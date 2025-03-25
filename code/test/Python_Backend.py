import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import requests
from bs4 import BeautifulSoup
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

# Connect with MongoDB
mongodb_url = 'mongodb+srv://<username>:<password>@<cluster>.mongodb.net'

try:
    # Find the database and collection folders
    client = pymongo.MongoClient(mongodb_url)
    db = client.get_database('test')
    collection = db.get_collection('users')
    results = collection.find({})

    # Convert MongoDB results to a Python list
    json_data = [result for result in results]

    # User profiling based on actions and time spent
    def profile_user(actions):
        element_counter = {}
        for action in actions:
            action_type = action["action"]
            if action_type in element_counter:
                element_counter[action_type] += 1
            else:
                element_counter[action_type] = 1

        user_type = max(element_counter, key=element_counter.get)

        last_time = actions[-1]["timeSpent"]
        if last_time < 20000:
            user_type += " impatient"
        else:
            user_type += " patient"

        return user_type

    # Classify users and assign names
    for item in json_data:
        click_types = profile_user(item["actions"])

        if "purchase" in click_types:
            item["click_types"] = "buyer"
        elif "informative" in click_types:
            item["click_types"] = "reader"
        elif "download" in click_types:
            item["click_types"] = "downloader"
        elif "internal-link" in click_types:
            item["click_types"] = "explorer"
        elif "external-link" in click_types:
            item["click_types"] = "referrer"
        else:
            item["click_types"] = click_types

        last_time = item["actions"][-1]["timeSpent"]
        if last_time < 20000:
            item["time_type"] = "impatient"
        else:
            item["time_type"] = "patient"

    # Create a feature matrix using CountVectorizer
    corpus = [item["time_type"] + " - " + item["click_types"] for item in json_data]  # Reverse order
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    # Apply the K-Means algorithm to cluster users into n clusters
    n_clusters = len(set(corpus))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X)

    # Assign clusters to each user
    for i, item in enumerate(json_data):
        item["cluster"] = kmeans.labels_[i]

        # Print the feature combination for each user
        print(f"User: {item['_id']} - Features: {item['time_type']} {item['click_types']}")

    # Get the most frequent words in documents for each cluster
    cluster_words = {}
    for cluster_id in range(n_clusters):
        cluster_docs = [corpus[i] for i, item in enumerate(json_data) if item["cluster"] == cluster_id]
        cluster_vectorizer = CountVectorizer()
        X_cluster = cluster_vectorizer.fit_transform(cluster_docs)
        word_freqs = Counter(cluster_vectorizer.get_feature_names_out())
        cluster_words[cluster_id] = word_freqs.most_common(2)

    # Examples of training data for each cluster (without matching keys)
    cluster_data = {
        "patient buyer": [
            "Patient buyer: I thoroughly research product reviews and compare prices before making a purchase decision.",
            "I take my time to read user feedback and consider the long-term benefits before buying a product.",
            "Careful analysis of features and customer experiences guide my buying choices.",
            "I prioritize understanding the value a product brings to make informed purchasing decisions.",
            "Patiently gathering information helps me avoid impulsive buys and ensures satisfaction."
        ],
        "impatient buyer": [
            "Impatient buyer: I make quick purchase decisions based on initial impressions and recommendations.",
            "I trust my instincts and prefer to buy products without spending too much time on research.",
            "Impulsive buying gives me a sense of instant gratification, even if I haven't fully explored options.",
            "I value convenience and rapid decision-making when it comes to making purchases.",
            "Quick purchases align with my busy lifestyle and preference for immediate results."
        ],
        "patient reader": [
            "Patient reader: I thoroughly delve into articles, absorbing every detail to gain comprehensive insights.",
            "Taking my time to understand complex topics enhances my learning experience.",
            "In-depth reading allows me to grasp different viewpoints and connect ideas effectively.",
            "Patiently absorbing content helps me form well-rounded perspectives and informed opinions.",
            "I enjoy the process of slowly unraveling information, leading to deeper understanding."
        ],
        "impatient reader": [
            "Impatient reader: I skim through articles, focusing on key points to quickly extract valuable information.",
            "Rapid reading helps me cover more content and gather main ideas in a short amount of time.",
            "I tend to skip unnecessary details and jump to the most relevant sections.",
            "Impulsive reading style enables me to swiftly assess the relevance of material to my interests.",
            "Quick information absorption suits my fast-paced lifestyle and multitasking habits."
        ],
        "patient downloader": [
            "Patient downloader: I meticulously download resources for thorough evaluation before implementation.",
            "Taking my time to test and analyze tools ensures I choose the best fit for my needs.",
            "Careful downloading and testing lead to better-informed decisions and successful outcomes.",
            "I prefer a methodical approach to downloading resources, focusing on quality over quantity.",
            "Patiently assessing resources before implementation minimizes errors and maximizes benefits."
        ],
        "impatient downloader": [
            "Impatient downloader: I rapidly download resources to address immediate needs and challenges.",
            "Quick access to tools helps me make swift progress without extensive preparation.",
            "I prioritize speed and efficiency when downloading resources for urgent tasks.",
            "Impulsive downloading suits my fast-paced work style and the need for quick solutions.",
            "Rapid resource acquisition enables me to respond promptly to changing circumstances."
        ],
        "patient explorer": [
            "Patient explorer: I take my time navigating different website sections to reveal hidden gems of information.",
            "Delving deep into a website's content allows me to discover valuable insights and unique perspectives.",
            "Careful exploration helps me make connections between different topics and enhance my knowledge.",
            "I enjoy immersing myself in the details of a website, discovering new layers of understanding.",
            "Patiently traversing through various sections enables me to create a comprehensive mental map of the website's content."
        ],
        "impatient explorer": [
            "Impatient explorer: I rapidly jump between website sections to quickly find the information I need.",
            "I prioritize speed and directness in my online exploration, focusing on my immediate interests.",
            "Quick navigation helps me extract key information without getting bogged down by irrelevant details.",
            "Impulsive exploration allows me to quickly assess the relevance of different website sections.",
            "Rapidly browsing through website content aligns with my preference for efficient information retrieval."
        ],
        "patient referrer": [
            "Patient referrer: I carefully curate and share links to high-quality websites that offer valuable insights.",
            "I take my time to evaluate the content and credibility of websites before recommending them to others.",
            "Thoroughly reviewing websites ensures that I share information that aligns with my standards.",
            "I prioritize sharing links that provide in-depth perspectives and contribute positively to discussions.",
            "Patiently selecting valuable resources helps me maintain a reputation for offering trusted recommendations."
        ],
        "impatient referrer": [
            "Impatient referrer: I quickly share interesting links with my network, sparking immediate engagement.",
            "I prioritize swift sharing of links that catch my attention, even if I haven't fully explored the content.",
            "Impulsive sharing helps me initiate discussions and capture the latest trends in real time.",
            "Quick link sharing aligns with my desire to keep my network informed and engaged.",
            "Rapidly spreading interesting content allows me to stay connected and contribute to ongoing conversations."
        ]
    }

    # Create names for clusters based on unique feature combinations of each user
    cluster_names = {}
    for cluster_id in range(n_clusters):
        users_in_cluster = [item for item in json_data if item["cluster"] == cluster_id]
        time_types = set([user['time_type'] for user in users_in_cluster])
        click_types = set([user['click_types'] for user in users_in_cluster])  # Reverse order

        cluster_name = ", ".join(time_types) + " " + ", ".join(click_types)  # Reverse order
        cluster_names[cluster_id] = cluster_name

    # Apply PCA to reduce dimensionality before visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())

    # Create a scatter plot to visualize clusters
    plt.figure(figsize=(12, 8))  # Adjust figure size

    # Get the list of clusters for all users
    cluster_labels = [item["cluster"] for item in json_data]

    # Define a function to obtain augmented examples
    def get_augmented_examples(example, num_augmentations=3):
        # Relevant keywords and words to exclude
        keywords = ["digital", "marketing", "online", "web", "internet", "development",
                    "user", "client", "customer", "interest", "website", "ecommerce",
                    "purchase"]
        exclude_words = ["i", "iodin", "iodine", "ampere", "angstrom", "I", "amp", "AN",
                         "A", "lots", "ME", "IN", "Maine", "In", "Inch"]

        synonyms = set()  # Use a set to store unique synonyms

        for word in example.split():
            synsets = wordnet.synsets(word)

            for syn in synsets:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")

                    if synonym != word and synonym not in synonyms and synonym not in exclude_words and not any(
                            keyword in synonym for keyword in keywords):
                        try:
                            similarity = wordnet.wup_similarity(wordnet.synset(word + '.n.01'),
                                                                wordnet.synset(synonym + '.n.01'))
                            if similarity and similarity > 0.6:  # Adjust similarity threshold as needed
                                synonyms.add(synonym)  # Add to the set instead of the list
                        except WordNetError:
                            pass

        # Query multiple online sources for synonyms
        sources = ["https://www.thesaurus.com/browse/"]
        for keyword in keywords:
            for source in sources:
                url = source + keyword
                response = requests.get(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    synonym_tags = soup.select(
                        ".css-1tu1tjh.e1wg9f5v4")  # Class of elements containing synonyms

                    for tag in synonym_tags:
                        synonym = tag.get_text().strip()
                        if synonym and synonym not in synonyms:
                            synonyms.add(synonym)  # Add to the set instead of the list

        # Convert the set into a list before applying random.sample
        synonyms_list = list(synonyms)
        return random.sample(synonyms_list, min(num_augmentations, len(synonyms_list)))

    # Generate augmented examples for each cluster
    augmented_cluster_data = {key: [] for key in cluster_data}
    for key, examples in cluster_data.items():
        augmented_examples = []
        for example in examples:
            augmented_examples.extend(get_augmented_examples(example))
        augmented_cluster_data[key] = augmented_examples

    # Combine original and augmented examples in the dictionary
    combined_examples = {key: cluster_data[key] + augmented_cluster_data[key] for key in cluster_data}

    # Print combined examples with reorganized synonyms only for automatically generated clusters
    for cluster_id in range(n_clusters):
        cluster_name = cluster_names[cluster_id]
        cluster_name_formatted = cluster_name  # Format cluster name in title case

        print(cluster_name_formatted + ":")

        if cluster_name in cluster_data and cluster_name in augmented_cluster_data:  # Check if examples are generated for this cluster
            examples = cluster_data[cluster_name] + augmented_cluster_data[cluster_name]

            # Print original examples and reorganized synonyms in the same order
            for example in examples:
                print("  - " + example)

    # Generate random positions to scatter points within each cluster (modified)
    np.random.seed(42)
    cluster_offsets_x = np.random.uniform(low=-0.15, high=0.15, size=len(json_data))
    cluster_offsets_y = np.random.uniform(low=-0.15, high=0.15, size=len(json_data))

    # Adjust point size and separation
    point_size = 150  # Point size
    separation_factor = 2.0  # Separation factor between points

    # Create the scatter plot in the reduced PCA dimensions
    scatter = plt.scatter(X_pca[:, 0] + cluster_offsets_x * separation_factor,
                          X_pca[:, 1] + cluster_offsets_y * separation_factor,
                          s=point_size, c=cluster_labels, cmap='viridis', alpha=0.7)

    # Add user labels to the plot
    for i, item in enumerate(json_data):
        plt.annotate(item["_id"], (X_pca[i, 0] + cluster_offsets_x[i] * separation_factor,
                                   X_pca[i, 1] + cluster_offsets_y[i] * separation_factor),
                     fontsize=9, ha='center', va='center', color='black', alpha=0.7, rotation=45)

    # Add cluster labels to the plot
    handles, labels = scatter.legend_elements(num=n_clusters)
    legend_labels = [f"{cluster_names[i]}" for i in range(n_clusters)]  # Use cluster name only
    plt.legend(handles, legend_labels, loc='upper right', title='Clusters')

    plt.xlabel('Dimension 1 (PCA)')
    plt.ylabel('Dimension 2 (PCA)')
    plt.title('User Profile Clustering (PCA)')
    plt.tight_layout()

    plt.show()

# Check successful connection to database
except pymongo.errors.ConnectionFailure as e:
    print(f'MongoDB connection error: {e}')
else:
    print('Data fetched successfully from MongoDB')
