# Pipeline de generation d'articles GEO/GSO

Ce projet est un mini-pipeline industrialisable pour generer des articles optimisés pour les moteurs de recherche génératifs (GEO). Le but c'est de passer d'un simple sujet (topic) à un article complet, scoré et pret à etre publié.

## Instalation

1. On commence par instaler les dépendances. J'ai utilisé `pip` pour ca :

   ```bash
   pip install -r requirements.txt
   ```

2. Il faut configurer le fichier `.env` à la racine. C'est la que tu mets tes clés API pour que LiteLLM puisse bosser :
   ```text
   LLM_MODEL=mistral/mistral-small-latest
   MISTRAL_API_KEY=ta_cle_ici
   # Ou pour Gemini
   GEMINI_API_KEY=ta_cle_ici
   ```

## Comment lancer le script

Pour generer les articles de test, tu peux lancer cette commande simple :

```bash
python generate.py --input topics.json --output ./out
```

Si tu as beaucoup de sujets et que tu veux aller plus vite, tu peux activer les workers en parallele :

```bash
python generate.py --input topics.json --output ./out --parallel --workers 4
```

## Les features implementées

J'ai essayé de couvrir tout ce qui était demandé dans l'exercice et meme un peu plus :

- **Generation structurée** : Chaque article respecte strictement le format demandé (H1, Meta desc, Intro, TOC, FAQ, Sources, Bloc auteur).
- **Scoring Qualité** : Un module calcul un score sur 100 basé sur 5 critères précis.
- **Anti-duplication** : On compare les articles entre eux pour eviter de publier 10 fois la meme chose.
- **Export multiple** : Le script sort du Markdown, du HTML (avec balises SEO) et un JSON tout pret pour une base de donnée.
- **Sauvegarde de secours** : Les articles sont sauvés sur le disque dès qu'ils sont generés, comme ca on perd rien si le script plante à la fin.
- **Mini-RAG** : Une base de doc locale peut etre injectée dans le prompt pour enrichir le contenu.
- **Publication WordPress** : Un client est pret pour envoyer les articles direct sur un site WP via l'API REST.

## Choix techniques detaillés

### 1. Choix du LLM (LiteLLM)
J'ai choisi d'utiliser **LiteLLM** comme SDK. L'avantage c'est que le code est totalement agnostique du fournisseur. On a testé avec **MISTRAL** (mistral-small-latest) et **GEMINI** (gemini-2.0-flash) et ca marche nickel. Si demain tu veux passer sur Claude ou GPT-4, t'as juste à changer une ligne dans ton `.env` sans toucher au code.

### 2. Module de Scoring (Scorer)
Le score final sur 100 est décomposé comme ca :
- **Structure (25 pts)** : On vérifie si toutes les balises sont la (H1, Meta, FAQ, etc) et si les longueurs sont respectées.
- **Lisibilité (20 pts)** : Basé sur l'indice de Flesch (via la librairie `textstat`). On penalise aussi les paragraphes trop longs.
- **Sources (20 pts)** : On compte le nombre de liens, leur diversité (domaines differents) et si c'est bien du HTTPS.
- **LLM-friendliness (20 pts)** : On check la densité d'entités nommées, la présence de listes et si l'intro est directe (pas de blabla inutile).
- **Duplication (15 pts)** : Ce score est ajusté par le module de déduplication.

### 3. Anti-duplication (Deduplicator)
Pour detecter les doublons, j'ai pas voulu refaire des appels API couteux. On utilise une approche locale avec **TF-IDF** et la **similarité cosinus** (via `scikit-learn`). 
- Le script compare chaque article généré avec tous les autres du lot.
- Si la similarité dépasse 0.85 (configurable), l'article est flaggé et son score baisse.
- C'est très rapide et ca tourne en local sur ta machine.

### 4. Robustesse et Multiprocessing
Le pipeline est paré pour la prod :
- **Retries** : Utilisation de `tenacity` pour gerer les echecs d'appels LLM (rate limits, timeouts).
- **Parallélisation** : Le flag `--parallel` utilise `multiprocessing` (spawn) pour lancer plusieurs générations en meme temps, ce qui divise le temps total par le nombre de workers.
- **Validation** : J'utilise `Pydantic V2` pour forcer le LLM à sortir du JSON propre. Si le JSON est foiré, le script essaie de le reparer ou demande au LLM de recommencer.

## Limites et pistes pour la suite

- **Scaling** : Pour passer à 1000 articles, il faudrait utiliser Celery avec Redis au lieu du multiprocessing actuel.
- **Fact-checking** : Le scoring actuel est syntaxique, on pourrait ajouter une etape de verification des faits via une deuxieme passe de LLM.
- **Monitoring** : Ajouter un dashboard pour voir l'avancée des scores en direct.
- **CMS** : Ajouter d'autres connecteurs (Shopify, Ghost).
