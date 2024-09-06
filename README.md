# Chatbot 

Bienvenue dans l'application **Chatbot** ! Cette application, construite avec le framework **Streamlit**, permet d'interagir avec des modèles multilingues et des modèles spécialisés dans le résumé de textes juridiques. L'application offre également une fonctionnalité avancée de **Retrieval-Augmented Generation (RAG)** pour des réponses plus précises et documentées.

## Fonctionnalités

### 1. Modèles Multilingues
L'application propose deux modèles multilingues :
- **Mistral**
- **SaulM**

Vous pouvez utiliser ces modèles de deux manières :
- **Utilisation simple** : Vous pouvez directement interagir avec des modèles multilingues pour générer des réponses.
- **Option RAG** : Vous avez également la possibilité d'utiliser la méthode **RAG** pour enrichir les réponses en intégrant des sources externes. Voici les étapes :
  1. **Chargement de fichiers** : Importez des documents pour constituer la base de données d'informations.
  2. **Text Splitter** : Sélectionnez une méthode de découpage du texte pour organiser les informations (différentes méthodes sont disponibles).
  3. **Sélection de l'Embedder** : Choisissez un modèle d'embedder pour représenter les textes sous forme de vecteurs.
  4. **Choix du Top-K** : Définissez le nombre de réponses les plus pertinentes à afficher.

### 2. Modèles de Résumé de Textes Juridiques
L'application offre plusieurs modèles spécialisés dans le résumé de textes juridiques :

- RoBerta-BART-Fixed
- RoBerta-BART-Dependent
- LongFormer-BART
- T5-EUR

Vous n'avez qu'à sélectionner l'un de ces modèles et l'application générera automatiquement un résumé du texte fourni.

## Prérequis

- Python 3.8+
- **Streamlit**
- **Langchain**
- **Transformers**
- Autres dépendances listées dans le fichier `requirements.txt`

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/amina-thioune/legal_chatbot/application.git
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application Streamlit :
   ```bash
   streamlit run main.py
   ```

## Utilisation

Une fois l'application démarrée, vous serez invité à choisir entre deux options principales :
- **Modèles multilingues** : Utilisez directement (Base) ou avec l'option **RAG**.
- **Modèles de résumé juridique** : Sélectionnez un modèle et résumez vos textes.



## Organisation des Répertoires

- Le répertoire `inferences` contient les scripts permettant d'interagir avec les modèles.
- Le répertoire `notebooks` regroupe les codes d'évaluation des modèles, l'implémentation de la méthode **RAG**, ainsi qu'un fichier détaillant les différentes techniques de découpage (**splitter**) et d'intégration (**embedder**).
- Le répertoire `presentation` inclut les supports de la soutenance, une vidéo de démonstration de l'application, ainsi qu'une présentation des différents articles étudiés.
- Le répertoire `rapport de stage` contient le rapport de stage.

