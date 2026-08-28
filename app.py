import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import chromadb
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

# 1. CONFIGURACIÓN DE LA PÁGINA WEB
st.set_page_config(
    page_title="AI E-Commerce Explorer (Free)", page_icon="🛍️", layout="wide"
)

st.title("🛍️ Buscador Semántico Gratuito y Segmentación de Reseñas")
st.markdown(
    "Esta aplicación es **100% Open Source**. Utiliza modelos locales de Hugging Face y ChromaDB sin necesidad de API Keys ni OpenAI."
)

# 2. CARGA DEL MODELO DE EMBEDDINGS GRATUITO


@st.cache_resource
def load_free_embedding_model():
    # Modelo ligero y veloz para correr de forma eficiente en los servidores gratuitos de Streamlit
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_free_embedding_model()


# 3. CARGA Y LIMPIEZA DE DATOS
@st.cache_data
def load_data():
    # Lee el archivo 'Reviews.csv' 
    df = pd.read_csv("Reviews.csv")
    df = df.dropna(subset=["Review Text"]).reset_index(drop=True)
    # Tomamos una muestra de 100 filas para garantizar velocidad en la carga del mapa t-SNE
    return df.sample(n=100, random_state=42).reset_index(drop=True)


df_reviews = load_data()
lista_textos = df_reviews["Review Text"].tolist()
lista_ids = [str(i) for i in df_reviews.index]


# 4. INICIALIZAR BASE DE DATOS VECTORIAL EN MEMORIA
@st.cache_resource
def get_vector_db():
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection("free_demo_collection")
    except:
        pass

    # Creamos la colección local
    collection = chroma_client.create_collection(name="free_demo_collection")

    # Calculamos los embeddings localmente en el servidor
    with st.spinner(
        "Inicializando base de datos vectorial con modelo Open Source..."
    ):
        vectores_calculados = embedding_model.encode(lista_textos).tolist()

        # Insertamos documentos y sus vectores correspondientes
        collection.add(
            documents=lista_textos,
            embeddings=vectores_calculados,
            ids=lista_ids,
        )
    return collection, vectores_calculados


db_collection, review_vectors = get_vector_db()


# 5. CREACIÓN DE LAS PESTAÑAS INTERACTIVAS
tab1, tab2 = st.tabs(
    ["🔍 Buscador Semántico en Vivo", "📊 Mapa Visual de Clientes (t-SNE)"]
)

with tab1:
    st.header("🤖 Motor de Búsqueda de Soporte")
    st.write(
        "Escribe un concepto o problema y el sistema encontrará los casos históricos más similares por su significado."
    )

    sugerencia = st.selectbox(
        "Ideas para buscar:",
        [
            "Absolutely wonderful - silky and sexy and comfortable",
            "The fabric was too thin and ripped easily",
            "It fits perfectly but the color is darker than the picture",
            "Escribe tu propia consulta...",
        ],
    )

    if sugerencia == "Escribe tu propia consulta...":
        query_usuario = st.text_input(
            "Ingresa tu búsqueda:", value="Looking for a summer dress"
        )
    else:
        query_usuario = sugerencia

    if st.button("Buscar coincidencias semánticas"):
        with st.spinner("Buscando en la base de datos vectorial..."):
            # Generamos el embedding de la consulta del usuario
            query_vector = embedding_model.encode([query_usuario]).tolist()

            # Consultamos la base de datos vectorial de ChromaDB
            results = db_collection.query(
                query_embeddings=query_vector, n_results=3
            )
            documentos_encontrados = results["documents"][0]

            st.success("¡Coincidencias encontradas!")
            for idx, doc in enumerate(documentos_encontrados, start=1):
                with st.chat_message("user"):
                    st.write(f"**Caso #{idx}:** {doc}")

with tab2:
    st.header("🗺️ Agrupación Semántica Automática")
    st.write(
        "Aquí puedes ver cómo la Inteligencia Artificial identifica patrones y agrupa las reseñas automáticamente por su significado real."
    )

    if st.button("Generar Mapa t-SNE con Categorías"):
        with st.spinner("Procesando clústeres... (Esto tomará unos segundos)"):

            # Temas fijos de clasificación
            temas_fijos = ["Quality", "Fit", "Style", "Comfort"]

            # Obtener embeddings de las categorías fijas
            cat_embeddings = embedding_model.encode(temas_fijos).tolist()

            # Clasificar dinámicamente cada reseña de la muestra para asignarle un color
            feedback_categories = []
            for text_emb in review_vectors:
                dists = [
                    distance.cosine(text_emb, cat_emb)
                    for cat_emb in cat_embeddings
                ]
                feedback_categories.append(temas_fijos[np.argmin(dists)])

            # Configurar y entrenar el modelo t-SNE de forma estable con inicialización de PCA
            tsne = TSNE(
                n_components=2, perplexity=10, random_state=42, init="pca"
            )
            tsne_results = tsne.fit_transform(np.array(review_vectors))

            # Empaquetar resultados en un DataFrame limpio para Seaborn
            df_plot = pd.DataFrame(
                {
                    "t-SNE Dimensión 1": tsne_results[:, 0],
                    "t-SNE Dimensión 2": tsne_results[:, 1],
                    "Categoría Semántica": feedback_categories,
                }
            )

            # Renderizar gráfico avanzado coloreado por categoría semántica
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df_plot,
                x="t-SNE Dimensión 1",
                y="t-SNE Dimensión 2",
                hue="Categoría Semántica",  # Activa la separación de colores lógicos
                palette="Dark2",  # Paleta profesional de alta visibilidad
                alpha=0.8,
                s=70,
                ax=ax,
            )

            ax.set_title(
                "Segmentación de Clientes: Reseñas de Ropa por Proximidad Semántica",
                fontsize=12,
            )
            ax.grid(True, linestyle="--", alpha=0.5)

            # Desplegar la figura de forma limpia en la interfaz web
            st.pyplot(fig)
