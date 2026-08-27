# C_dialogues — contenu conversationnel de la brique C

499 dialogues français + 196 code-switch, générés par `lfm2-generate-fr`
(Gemini), filtrés contre les benchmarks held-out (`lang_mirror`, `fr_s2s`).

**98 % des dialogues code-switch changent réellement de langue** — mesuré, parce
qu'un lot qui aurait seulement l'air bilingue n'entraînerait rien.

Versionné ici, et non dans `data/` : c'est du **texte généré**, l'entrée
reproductible de la synthèse de la brique A, au même titre que les
`benchmark/*/questions.jsonl`. Seuls l'audio et les poids restent hors de git.

Schéma : une ligne par dialogue, `meta.expected_lang` porte la langue que
l'assistant doit employer — celle du DERNIER tour utilisateur, pas du premier.
