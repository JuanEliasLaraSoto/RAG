# Mini RAG

Un RAG (Retrieval-Augmented Generation) mínimo en un solo archivo: `rag.py`.

1. **Carga y fragmenta** los `.md`/`.txt` de `docs/` en trozos de ~80 palabras.
2. **Recupera** los fragmentos más relevantes con BM25, escrito en Python puro y sin dependencias.
3. **Genera** la respuesta con Claude, usando solo esos fragmentos y citando la fuente.

## Uso

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python rag.py "¿Cuál es el planeta más grande?"
python rag.py                          # modo interactivo
python rag.py --solo-buscar "espresso" # solo recuperación, sin API
python rag.py -k 5 --docs mis_docs/    # otra carpeta, más fragmentos
```

Para usar tus propios documentos, ponlos en `docs/` (o pasa `--docs`).

## Ideas para crecer

- Cambiar BM25 por embeddings (p. ej. `sentence-transformers`) y búsqueda vectorial.
- Combinar ambos métodos (búsqueda híbrida).
- Guardar el índice en disco para no recalcularlo en cada ejecución.
