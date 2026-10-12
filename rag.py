"""Mini RAG: recupera fragmentos con BM25 y responde con Claude.

Uso:
    python rag.py "¿Cuál es el planeta más grande?"
    python rag.py                       # modo interactivo
    python rag.py --solo-buscar "café"  # solo recuperación, sin llamar a la API
"""

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MODELO = "claude-opus-5-5"
STOPWORDS = set(
    "a al como con de del el en es esta este fue la las lo los mas o para "
    "por que se su sus un una y cual cuales quien donde cuando cuanto".split()
)


@dataclass
class Fragmento:
    fuente: str
    texto: str


# ---------- 1. Carga y fragmentación ----------

def cargar_fragmentos(carpeta: Path, max_palabras: int = 80) -> list[Fragmento]:
    """Divide cada .md/.txt en párrafos y agrupa los cortos hasta ~max_palabras."""
    fragmentos = []
    for ruta in sorted(carpeta.glob("**/*")):
        if ruta.suffix not in {".md", ".txt"}:
            continue
        actual: list[str] = []
        for parrafo in re.split(r"\n\s*\n", ruta.read_text(encoding="utf-8")):
            parrafo = parrafo.strip()
            if not parrafo:
                continue
            if actual and len(" ".join(actual + [parrafo]).split()) > max_palabras:
                fragmentos.append(Fragmento(ruta.name, "\n".join(actual)))
                actual = []
            actual.append(parrafo)
        if actual:
            fragmentos.append(Fragmento(ruta.name, "\n".join(actual)))
    return fragmentos


# ---------- 2. Recuperación (BM25) ----------

def tokenizar(texto: str) -> list[str]:
    sin_tildes = unicodedata.normalize("NFKD", texto.lower())
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return [t for t in re.findall(r"\w+", sin_tildes) if t not in STOPWORDS]


class BM25:
    def __init__(self, fragmentos: list[Fragmento], k1: float = 1.5, b: float = 0.75):
        self.fragmentos = fragmentos
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenizar(f.texto)) for f in fragmentos]
        self.longitudes = [sum(d.values()) for d in self.docs]
        self.longitud_media = sum(self.longitudes) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[float, Fragmento]]:
        terminos = tokenizar(consulta)
        resultados = []
        for doc, largo, frag in zip(self.docs, self.longitudes, self.fragmentos):
            puntaje = 0.0
            for t in terminos:
                tf = doc.get(t, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * largo / self.longitud_media)
                    puntaje += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if puntaje > 0:
                resultados.append((puntaje, frag))
        resultados.sort(key=lambda x: x[0], reverse=True)
        return resultados[:k]


# ---------- 3. Generación con Claude ----------

def responder(pregunta: str, contexto: list[Fragmento]) -> str:
    import anthropic

    bloques = "\n\n".join(
        f'<fragmento id="{i}" fuente="{f.fuente}">\n{f.texto}\n</fragmento>'
        for i, f in enumerate(contexto, 1)
    )
    client = anthropic.Anthropic()
    respuesta = client.beta.messages.create(
        model=MODELO,
        max_tokens=16000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=(
            "Responde en español usando solo la información de los fragmentos. "
            "Cita la fuente entre corchetes, p. ej. [cafe.md]. Si los fragmentos "
            "no contienen la respuesta, dilo claramente."
        ),
        messages=[{
            "role": "user",
            "content": f"<contexto>\n{bloques}\n</contexto>\n\nPregunta: {pregunta}",
        }],
    )
    if respuesta.stop_reason == "refusal":
        return "(El modelo declinó responder esta pregunta.)"
    return "".join(b.text for b in respuesta.content if b.type == "text")


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RAG con BM25 + Claude")
    parser.add_argument("pregunta", nargs="?", help="pregunta (vacío = modo interactivo)")
    parser.add_argument("--docs", default=Path(__file__).parent / "docs", type=Path)
    parser.add_argument("-k", type=int, default=3, help="fragmentos a recuperar")
    parser.add_argument("--solo-buscar", action="store_true", help="no llamar a Claude")
    args = parser.parse_args()

    indice = BM25(cargar_fragmentos(args.docs))
    print(f"Indexados {len(indice.fragmentos)} fragmentos de {args.docs}\n")

    def procesar(pregunta: str) -> None:
        resultados = indice.buscar(pregunta, args.k)
        if not resultados:
            print("No encontré fragmentos relevantes.\n")
            return
        for puntaje, frag in resultados:
            print(f"  [{puntaje:.2f}] {frag.fuente}: {frag.texto[:70]}...")
        if not args.solo_buscar:
            try:
                print("\n" + responder(pregunta, [f for _, f in resultados]))
            except TypeError as e:  # el SDK lanza TypeError si no hay credenciales
                if "authentication" not in str(e):
                    raise
                print("\nFalta ANTHROPIC_API_KEY. Expórtala o usa --solo-buscar.")
        print()

    if args.pregunta:
        procesar(args.pregunta)
        return
    while True:
        try:
            pregunta = input("Pregunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if pregunta:
            procesar(pregunta)


if __name__ == "__main__":
    sys.exit(main())
