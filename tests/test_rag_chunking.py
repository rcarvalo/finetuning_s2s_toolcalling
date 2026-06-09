from s2s_toolcalling.rag.ingest import chunk_text


def test_short_text_single_chunk():
    chunks = list(chunk_text("Bonjour.\n\nPetit document.", chunk_size=100, overlap=10))
    assert chunks == ["Bonjour.\n\nPetit document."]


def test_paragraphs_grouped_until_size():
    paras = [f"Paragraphe numéro {i} avec un peu de contenu." for i in range(10)]
    text = "\n\n".join(paras)
    chunks = list(chunk_text(text, chunk_size=120, overlap=20))
    assert len(chunks) > 1
    assert all(chunks)
    # tout le contenu est couvert
    joined = " ".join(chunks)
    for i in range(10):
        assert f"Paragraphe numéro {i}" in joined


def test_long_paragraph_hard_split():
    text = "x" * 2500
    chunks = list(chunk_text(text, chunk_size=800, overlap=100))
    assert all(chunks)
    assert sum(len(c) for c in chunks) >= 2500
