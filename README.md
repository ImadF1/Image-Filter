# Image Filter Studio

Application Python de traitement d'image avec interface Streamlit.

Le projet permet de charger une image, appliquer plusieurs filtres visuels en temps reel, comparer le resultat avec l'original, puis exporter l'image modifiee.

## Fonctionnalites

- conversion en niveaux de gris
- detection de contours
- flou gaussien
- effet sepia
- effet vintage
- ajustement de la luminosite
- ajustement du contraste
- apercu avant / apres
- export en PNG

## Stack technique

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit

## Structure du projet

```text
image-filter-app/
├── app.py
├── filters.py
├── requirements.txt
├── README.md
└── tests/
    └── test_filters.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

Puis ouvrir l'URL affichee par Streamlit dans le navigateur.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Cas d'usage

- retouche rapide d'images
- generation d'effets visuels simples
- mini projet de computer vision
- demonstration OpenCV avec interface web legere

## Ameliorations possibles

- ajout d'autres filtres
- historique des modifications
- traitement par lot
- sauvegarde dans plusieurs formats
