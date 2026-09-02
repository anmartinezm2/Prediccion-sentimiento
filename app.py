# ============================================================
# APLICACIÓN WEB - CLASIFICACIÓN DE SENTIMIENTOS
# ============================================================
#
# Modelo:
# Regresión Logística
#
# Archivos utilizados:
# modelo.pkl
# vectorizador.pkl
# tfidf.pkl
#
# Categorías:
#
# peaceful  -> Tranquilo
# mad       -> Enojado
# powerful  -> Poderoso
# sad       -> Triste
# joyful    -> Alegre
# scared    -> Asustado
# ============================================================


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import string
import nltk

from nltk.corpus import stopwords


# ============================================================
# 2. CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Analizador de Sentimientos",
    page_icon="😊",
    layout="centered"
)


# ============================================================
# 3. TÍTULO DE LA APLICACIÓN
# ============================================================

st.title("🤖 Analizador de Sentimientos")

st.write(
    "Escribe un texto y el modelo de Regresión Logística "
    "intentará identificar su sentimiento."
)

st.divider()


# ============================================================
# 4. TRADUCCIÓN DE LOS SENTIMIENTOS
# ============================================================

# El modelo trabaja internamente con las categorías
# originales en inglés.
#
# Aquí las traducimos para mostrarlas al usuario.

traduccion_sentimientos = {

    'peaceful': 'Tranquilo',
    'mad': 'Enojado',
    'powerful': 'Poderoso',
    'sad': 'Triste',
    'joyful': 'Alegre',
    'scared': 'Asustado'

}


# Emojis para cada sentimiento

emojis_sentimientos = {

    'peaceful': '😌',
    'mad': '😡',
    'powerful': '💪',
    'sad': '😢',
    'joyful': '😊',
    'scared': '😨'

}


# ============================================================
# 5. CARGAR STOPWORDS EN ESPAÑOL
# ============================================================

# Descargar stopwords si todavía no están disponibles.

nltk.download(
    'stopwords',
    quiet=True
)


# Obtener stopwords en español.

STOPWORDS_ES = set(
    stopwords.words('spanish')
)


# ============================================================
# 6. CONSERVAR PALABRAS IMPORTANTES
# ============================================================

# Estas palabras NO deben eliminarse porque pueden cambiar
# completamente el significado de una oración.
#
# Ejemplo:
#
# "Estoy feliz"
#
# "No estoy feliz"
#
# La palabra "no" es fundamental para distinguir ambas frases.

palabras_importantes = {

    'no',
    'nunca',
    'jamás',
    'nada',
    'nadie',
    'ningún',
    'ninguna',
    'ninguno',
    'ni',
    'tampoco',
    'sin'

}


# Eliminar estas palabras de la lista de stopwords.

STOPWORDS_ES = STOPWORDS_ES - palabras_importantes


# ============================================================
# 7. FUNCIÓN DE PREPROCESAMIENTO
# ============================================================

# IMPORTANTE:
#
# Esta función debe ser igual a la utilizada durante
# el entrenamiento del modelo en Google Colab.
#
# Si el texto nuevo se procesa de una manera diferente,
# las predicciones pueden cambiar.

def text_process(mess):

    # Convertir a texto

    mess = str(mess)


    # Convertir a minúsculas

    mess = mess.lower()


    # Eliminar signos de puntuación

    nopunc = [

        char

        for char in mess

        if char not in string.punctuation

    ]


    # Volver a unir los caracteres

    nopunc = ''.join(
        nopunc
    )


    # Separar las palabras

    palabras = nopunc.split()


    # Eliminar stopwords

    palabras_limpias = [

        palabra

        for palabra in palabras

        if palabra not in STOPWORDS_ES

    ]


    # Volver a unir las palabras

    return ' '.join(
        palabras_limpias
    )


# ============================================================
# 8. CARGAR EL MODELO
# ============================================================

@st.cache_resource
def cargar_modelos():

    modelo = joblib.load(
        'modelo.pkl'
    )

    vectorizador = joblib.load(
        'vectorizador.pkl'
    )

    tfidf = joblib.load(
        'tfidf.pkl'
    )

    return modelo, vectorizador, tfidf


# Cargar los archivos

try:

    modelo, vectorizador, tfidf = cargar_modelos()

except Exception as e:

    st.error(
        "No se pudieron cargar los archivos del modelo."
    )

    st.info(
        "Verifica que modelo.pkl, vectorizador.pkl y "
        "tfidf.pkl estén en la misma carpeta que app.py."
    )

    st.stop()


# ============================================================
# 9. CAMPO PARA ESCRIBIR EL TEXTO
# ============================================================

st.subheader("✍️ Escribe un texto")


texto = st.text_area(

    "Texto a analizar:",

    placeholder=(
        "Ejemplo: Hoy estoy muy feliz porque "
        "aprobé mi examen..."
    ),

    height=150

)


# ============================================================
# 10. REALIZAR LA PREDICCIÓN
# ============================================================

if texto.strip() != "":

    # --------------------------------------------------------
    # Limpiar el texto
    # --------------------------------------------------------

    texto_limpio = text_process(
        texto
    )


    # --------------------------------------------------------
    # Convertir texto a números
    # --------------------------------------------------------

    texto_dtm = vectorizador.transform(
        [texto_limpio]
    )


    # --------------------------------------------------------
    # Aplicar TF-IDF
    # --------------------------------------------------------

    texto_tfidf = tfidf.transform(
        texto_dtm
    )


    # --------------------------------------------------------
    # Realizar predicción
    # --------------------------------------------------------

    prediccion = modelo.predict(
        texto_tfidf
    )


    sentimiento = prediccion[0]


    # --------------------------------------------------------
    # Obtener probabilidades
    # --------------------------------------------------------

    probabilidades = modelo.predict_proba(
        texto_tfidf
    )[0]


    # Crear DataFrame con probabilidades

    resultado_prob = pd.DataFrame({

        'sentimiento': modelo.classes_,

        'probabilidad': probabilidades

    })


    # Ordenar de mayor a menor

    resultado_prob = resultado_prob.sort_values(

        by='probabilidad',

        ascending=False

    )


    # ========================================================
    # 11. MOSTRAR RESULTADO PRINCIPAL
    # ========================================================

    st.divider()

    st.subheader("🔮 Resultado")


    nombre_espanol = traduccion_sentimientos.get(

        sentimiento,

        sentimiento

    )


    emoji = emojis_sentimientos.get(

        sentimiento,

        '🔮'

    )


    st.success(

        f"{emoji} Sentimiento detectado: "
        f"**{nombre_espanol}**"

    )


    # Obtener confianza

    confianza = probabilidades.max() * 100


    st.metric(

        "Confianza aproximada",

        f"{confianza:.2f}%"

    )


    # ========================================================
    # 12. MOSTRAR TEXTO ANALIZADO
    # ========================================================

    st.subheader("📝 Texto analizado")

    st.write(
        texto
    )


    # ========================================================
    # 13. MOSTRAR TEXTO PROCESADO
    # ========================================================

    with st.expander(
        "🔍 Ver texto después del preprocesamiento"
    ):

        st.write(
            texto_limpio
        )


    # ========================================================
    # 14. MOSTRAR PROBABILIDADES
    # ========================================================

    st.subheader(
        "📊 Probabilidades por sentimiento"
    )


    # Crear tabla para mostrar al usuario

    tabla_probabilidades = pd.DataFrame({

        'Sentimiento': [

            traduccion_sentimientos.get(

                sentimiento,

                sentimiento

            )

            for sentimiento
            in resultado_prob['sentimiento']

        ],

        'Probabilidad (%)': [

            prob * 100

            for prob
            in resultado_prob['probabilidad']

        ]

    })


    # Redondear

    tabla_probabilidades[
        'Probabilidad (%)'
    ] = tabla_probabilidades[
        'Probabilidad (%)'
    ].round(2)


    # Mostrar tabla

    st.dataframe(

        tabla_probabilidades,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # 15. GRÁFICA DE PROBABILIDADES
    # ========================================================

    st.subheader(
        "📈 Distribución de probabilidades"
    )


    # Crear una copia para la gráfica

    grafica = tabla_probabilidades.copy()


    # Usar el sentimiento como índice

    grafica = grafica.set_index(
        'Sentimiento'
    )


    st.bar_chart(
        grafica['Probabilidad (%)']
    )


    # ========================================================
    # 16. INTERPRETACIÓN
    # ========================================================

    st.subheader(
        "💡 Interpretación"
    )


    if confianza >= 70:

        st.write(

            f"El modelo identifica el texto principalmente "
            f"como **{nombre_espanol}**, con una probabilidad "
            f"aproximada del **{confianza:.2f}%**."

        )

    elif confianza >= 50:

        st.warning(

            f"El modelo se inclina hacia **{nombre_espanol}**, "
            f"pero la predicción presenta cierta incertidumbre "
            f"({confianza:.2f}%)."

        )

    else:

        st.warning(

            f"El modelo predice **{nombre_espanol}**, pero "
            f"la confianza es baja ({confianza:.2f}%). "
            f"El texto podría compartir características "
            f"con varias categorías."

        )


# ============================================================
# 17. INFORMACIÓN DEL MODELO
# ============================================================

st.divider()

with st.expander(
    "ℹ️ Información del modelo"
):

    st.write(
        "Modelo utilizado: **Regresión Logística**"
    )

    st.write(
        "Vectorización: **CountVectorizer**"
    )

    st.write(
        "Transformación: **TF-IDF**"
    )

    st.write(
        "Categorías: **6 sentimientos**"
    )

    st.write(
        "Idioma de los textos: **Español**"
    )
