# 🛍️ AI E-Commerce Search & Segmentation (100% Open-Source)

¡Bienvenido! Este es un proyecto de **Búsqueda Semántica** y **Segmentación de Clientes** enfocado en el comercio electrónico. A diferencia de los buscadores tradicionales basados en palabras clave exactas, este sistema entiende el significado y contexto del feedback de los usuarios.

La aplicación está construida completamente con tecnologías de código abierto (*Open-Source*), lo que significa que corre de forma local y **gratuita** sin depender de APIs de pago como OpenAI.

🚀 **[PROBAR LA APP EN VIVO AQUÍ](AQUÍ_PEGAS_TU_ENLACE_DE_STREAMLIT_CUANDO_LO_TENGAS)**

---

## 🧠 Características Principales

* **Buscador Semántico en Tiempo Real:** Permite a los equipos de soporte encontrar casos históricos o reseñas similares basándose en la intención del texto (ej. tolera sinónimos y errores conceptuales).
* **Base de Datos Vectorial Local:** Utiliza **ChromaDB** para indexar y consultar embeddings vectoriales en milisegundos.
* **Embeddings de Código Abierto:** Implementa el modelo `all-MiniLM-L6-v2` de *Sentence-Transformers* (Hugging Face) para transformar texto a vectores de 384 dimensiones de manera gratuita.
* **Visualización Avanzada (t-SNE):** Reduce las dimensiones de los vectores de alta densidad a un plano 2D para identificar patrones lógicos en los intereses de los clientes.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.10+**
* **Streamlit** (Interfaz web interactiva)
* **ChromaDB** (Almacenamiento y base de datos vectorial)
* **Sentence-Transformers / Hugging Face** (Modelos de lenguaje local)
* **Scikit-Learn** (Algoritmo t-SNE para reducción de dimensiones)
* **Pandas & NumPy** (Manipulación y limpieza de datos)
* **Matplotlib & Seaborn** (Visualización estática de datos)

---

## 📦 Instalación y Uso Local

Si deseas clonar este repositorio y correrlo en tu propia computadora, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com
   cd TU_REPOSITORIO
   ```

2. **Crear e inicializar un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

3. **Instalar las dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación con Streamlit:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset Utilizado
El proyecto utiliza el popular dataset **Women's Clothing E-Commerce Reviews** disponible en Kaggle, el cual contiene más de 23,000 reseñas reales de clientes anonimizadas.
