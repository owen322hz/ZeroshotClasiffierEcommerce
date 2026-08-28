import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import chromadb
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

# 1. CONFIGURACIÓN DE LA PÁGINA WEB
st.set_page_config(
    page_title="AI E-Commerce Explorer (Max)", page_icon="🛍️", layout="wide"
)

st.title("🛍️ Buscador Semántico Gratuito y Segmentación de Reseñas")
st.markdown(
    "Esta aplicación es **100% Open Source**. Utiliza modelos locales de Hugging Face y ChromaDB sin necesidad de API Keys ni OpenAI."
)

# 2. CARGA DEL MODELO DE EMBEDDINGS GRATUITO
@st.cache_resource
def load_free_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_free_embedding_model()

# 3. CONTROL DINÁMICO (SLIDER) EN LA BARRA LATERAL
with st.sidebar:
    st.header("🎛️ Panel de Control")
    st.write("Ajusta el tamaño de la muestra para evaluar el comportamiento de la IA a diferentes escalas.")
    
    # Slider dinámico: Mínimo 50, Máximo 500, Valor por defecto 150, Pasos de 50 en 50
    tamano_muestra = st.slider(
        "Cantidad de reseñas a procesar:",
        min_value=50,
        max_value=500,
        value=150,
        step=50,
        help="Un tamaño mayor incrementa la precisión estadística pero requiere más cómputo."
    )
    
    st.info(f"Procesando actualmente: {tamano_muestra} reseñas.")

# 4. CARGA Y LIMPIEZA DE DATOS (Ajustado dinámicamente por el Slider)
@st.cache_data
def load_data(sample_size):
    df = pd.read_csv("Reviews.csv")
    df = df.dropna(subset=["Review Text", "Recommended IND"]).reset_index(drop=True)
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

df_reviews = load_data(tamano_muestra)
lista_textos = df_reviews["Review Text"].tolist()
lista_ids = [str(i) for i in df_reviews.index]


# 5. INICIALIZAR BASE DE DATOS VECTORIAL EN MEMORIA (Dinámica y sin errores de caché)
def get_vector_db(texts, ids, _model):
    chroma_client = chromadb.Client()
    
    # Nombre de colección único dinámico basado en la cantidad de elementos
    nombre_coleccion = f"free_demo_collection_{len(texts)}"
    
    try:
        chroma_client.delete_collection(nombre_coleccion)
    except:
        pass

    collection = chroma_client.create_collection(name=nombre_coleccion)

    with st.spinner("Inicializando base de datos vectorial con modelo Open Source..."):
        vectores_calculados = _model.encode(texts).tolist()
        collection.add(
            documents=texts,
            embeddings=vectores_calculados,
            ids=ids,
        )
    return collection, vectores_calculados

# Se ejecuta fresco cada vez que cambia la muestra del Slider para evitar el NotFoundError
db_collection, review_vectors = get_vector_db(lista_textos, lista_ids, embedding_model)


# 6. CREACIÓN DE LAS PESTAÑAS INTERACTIVAS
tab1, tab2, tab3 = st.tabs(
    ["🔍 Buscador Semántico en Vivo", "📊 Mapa Visual (t-SNE)", "📈 Métricas de Evaluación"]
)

with tab1:
    st.header("🤖 Motor de Búsqueda de Soporte")
    st.write("Escribe un concepto o problema y el sistema encontrará los casos históricos más similares dentro del volumen seleccionado.")

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
        query_usuario = st.text_input("Ingresa tu búsqueda:", value="Looking for a summer dress")
    else:
        query_usuario = sugerencia

    if st.button("Buscar coincidencias semánticas"):
        with st.spinner("Buscando en la base de datos vectorial..."):
            query_vector = embedding_model.encode([query_usuario]).tolist()
            results = db_collection.query(query_embeddings=query_vector, n_results=3)
            documentos_encontrados = results["documents"]

            st.success("¡Coincidencias encontradas!")
            for idx, doc in enumerate(documentos_encontrados, start=1):
                with st.chat_message("user"):
                    st.write(f"**Caso #{idx}:** {doc}")

with tab2:
    st.header("🗺️ Agrupación Semántica Automática")
    st.write(f"Visualización geométrica interactiva de las {tamano_muestra} opiniones de los clientes.")

    if st.button("Generar Mapa t-SNE con Categorías"):
        with st.spinner("Procesando clústeres..."):
            temas_fijos = ["Quality", "Fit", "Style", "Comfort"]
            cat_embeddings = embedding_model.encode(temas_fijos).tolist()

            feedback_categories = []
            for text_emb in review_vectors:
                dists = [distance.cosine(text_emb, cat_emb) for cat_emb in cat_embeddings]
                feedback_categories.append(temas_fijos[np.argmin(dists)])

            # Ajustamos la perplejidad de forma segura basada en el tamaño del vector actual
            dinamic_perplexity = min(10, max(2, len(review_vectors) // 10))
            tsne = TSNE(n_components=2, perplexity=dinamic_perplexity, random_state=42, init="pca")
            tsne_results = tsne.fit_transform(np.array(review_vectors))

            df_plot = pd.DataFrame({
                "t-SNE Dimensión 1": tsne_results[:, 0],
                "t-SNE Dimensión 2": tsne_results[:, 1],
                "Categoría Semántica": feedback_categories,
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df_plot, x="t-SNE Dimensión 1", y="t-SNE Dimensión 2",
                hue="Categoría Semántica", palette="Dark2", alpha=0.8, s=70, ax=ax
            )
            ax.set_title("Segmentación de Reseñas por Proximidad Semántica", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

with tab3:
    st.header("📈 Evaluación de Rendimiento Semántico")
    st.write(f"Evaluación estadística del clasificador Zero-Shot sobre la muestra actual de {tamano_muestra} elementos.")

    if st.button("Calcular Métricas de Calidad"):
        with st.spinner("Ejecutando pruebas estadísticas..."):
            clases_sentimiento = [
                "This clothing item is terrible, worst purchase, bad quality, completely disappointed",
                "This clothing item is absolutely amazing, highly recommended, great fit, love it"
            ]
            clases_embeddings = embedding_model.encode(clases_sentimiento).tolist()

            predicciones = []
            for text_emb in review_vectors:
                dists = [distance.cosine(text_emb, c_emb) for c_emb in clases_embeddings]
                predicciones.append(np.argmin(dists))

            valores_reales = df_reviews["Recommended IND"].astype(int).tolist()

            cm = confusion_matrix(valores_reales, predicciones)
            report = classification_report(valores_reales, predicciones, output_dict=True)
            accuracy = report['accuracy'] * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Exactitud General (Accuracy)", f"{accuracy:.2f}%")
            col2.metric("Precisión Positivos (Precision)", f"{report['1']['precision']*100:.2f}%")
            col3.metric("Sensibilidad Positivos (Recall)", f"{report['1']['recall']*100:.2f}%")

            st.write("---")
            
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                xticklabels=["Predicho: No Rec.", "Predicho: Sí Rec."],
                yticklabels=["Real: No Rec.", "Real: Sí Rec."]
            )
            ax_cm.set_title("Matriz de Confusión (Zero-Shot Sentiment vs Real)")
            st.pyplot(fig_cm)
            st.markdown(
                "💡 **Interpretación:** Al modificar la muestra con el Slider, verás cómo cambian los "
                "volúmenes absolutos dentro de los cuadrantes, evaluando la estabilidad del modelo geométrico."
            )
