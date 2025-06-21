# -*- coding: utf-8 -*-
"""app

Este script implementa un Sistema de Aprendizaje Personalizado para Flipped Classroom
utilizando Streamlit, Pandas, la API de Gemini para la generación dinámica de contenido
y Pinecone para Retrieval Augmented Generation (RAG).
Los niveles de conocimiento de los estudiantes se gestionan a través de un archivo CSV local,
el cual puede ser actualizado manualmente o mediante la carga de resultados de exámenes (CSV).
"""

import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import os
import time # Para reintentos en la generación de embeddings
from pinecone import Pinecone # Importar Pinecone
# Si vas a usar LangChain para dividir texto dentro de la app (no recomendado para este caso),
# o si quieres mantener la coherencia de imports, no es estrictamente necesario aquí
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuración de la API de Gemini y Pinecone ---
gemini_api_key = None
pinecone_api_key = None
pinecone_environment = None
pinecone_index_name = "aulainvertigarag" # Asegúrate que este sea el nombre de tu índice

try:
    # Intenta cargar desde Streamlit Secrets (preferido para despliegue)
    gemini_api_key = st.secrets["gemini_api_key"]
    pinecone_api_key = st.secrets["pinecone_api_key"]
    pinecone_environment = st.secrets["pinecone_environment"]
    # Si el nombre del índice también es un secreto
    pinecone_index_name = st.secrets.get("pinecone_index_name", "aulainvertigarag")
except AttributeError:
    # Si no estás en Streamlit Cloud o secrets.toml no está configurado, usa variables de entorno
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "aulainvertigarag")

# Validar que las claves estén configuradas
if not gemini_api_key:
    st.error(
        "Error: La clave de API de Gemini no ha sido configurada. "
        "Asegúrate de establecer 'gemini_api_key' en Streamlit Secrets o como variable de entorno (GEMINI_API_KEY)."
    )
    st.stop()

if not pinecone_api_key or not pinecone_environment:
    st.error(
        "Error: Las claves de API o el entorno de Pinecone no han sido configurados. "
        "Asegúrate de establecer 'pinecone_api_key' y 'pinecone_environment' en Streamlit Secrets o como variables de entorno."
    )
    st.stop()

# Configurar Gemini
genai.configure(api_key=gemini_api_key)

# CAMBIO IMPORTANTE AQUÍ: Gestionar la instancia del modelo con st.session_state
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
model = st.session_state.gemini_model # Usar la instancia del modelo desde session_state

EMBEDDING_MODEL = "models/embedding-001" # El mismo modelo que usaste para indexar

# Inicializar Pinecone
try:
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

    # Obtenemos la lista de índices y extraemos solo sus nombres
    existing_indexes_info = pc.list_indexes()
    existing_index_names = [idx_info['name'] for idx_info in existing_indexes_info]

    # Ahora sí, verificamos si el nombre del índice está en la lista de nombres
    if pinecone_index_name not in existing_index_names:
        st.error(f"Error: El índice de Pinecone '{pinecone_index_name}' no existe en tu entorno '{pinecone_environment}'. Por favor, verifica el nombre del índice o créalo.")
        st.stop()

    index = pc.Index(pinecone_index_name)
    st.success(f"Conexión a Pinecone establecida con el índice: **{pinecone_index_name}**.")
except Exception as e:
    st.error(f"Error al conectar con Pinecone: {e}. Por favor, verifica tu clave API, entorno y el nombre del índice.")
    st.stop()


# --- Definiciones Globales ---
conceptos = [
    "FC_DEFINICION",
    "FC_ROLES",
    "FC_TECNOLOGIA",
    "FC_APLICACION",
    "FC_BENEFICIOS",
]

nivel_map = {
    0.25: "Básico",
    0.55: "Intermedio",
    0.85: "Avanzado",
}

# Diccionario para nombres legibles de conceptos
conceptos_legibles = {
    "FC_DEFINICION": "la definición de Flipped Classroom",
    "FC_ROLES": "los roles del estudiante y el docente en Flipped Classroom",
    "FC_TECNOLOGIA": "la tecnología usada en Flipped Classroom",
    "FC_APLICACION": "cómo aplicar Flipped Classroom",
    "FC_BENEFICIOS": "los beneficios de Flipped Classroom",
}

# --- Cargar datos de estudiantes y citas ---
# Asegúrate de que estos archivos estén en la misma carpeta que app.py o accesible
try:
    estudiantes_df = pd.read_csv("estudiantes.csv")
except FileNotFoundError:
    st.error("Error: 'estudiantes.csv' no encontrado. Asegúrate de que esté en la misma carpeta.")
    st.stop()

citations_dict = {}
try:
    citation_data_path = "Fuentes Aula invertida.csv" # Asegúrate que este archivo esté accesible
    simplified_df = pd.read_csv(citation_data_path, encoding='latin1', delimiter=';')
    # Ajusta los nombres de las columnas según tu CSV si son diferentes
    simplified_df = simplified_df.rename(columns={
        simplified_df.columns[0]: "filename", # Asumimos que esta columna es el "cite_apa" corto o un ID que usaremos como clave
        simplified_df.columns[1]: "cite_apa",
        simplified_df.columns[2]: "reference_apa"
    })
    for _, row in simplified_df.iterrows():
        # Aquí, 'file_name' es el valor de la primera columna de tu CSV (ej. "Colomo-Magana et al., 2023")
        # Es CRÍTICO que los valores de 'cite_apa' y 'reference_apa' también se almacenen en los metadatos de Pinecone
        # con la clave exacta que usamos aquí ('cite_apa' y 'reference_apa').
        file_key_from_csv = str(row['filename']).strip() # Usamos la primera columna como clave del diccionario.
        citations_dict[file_key_from_csv] = {
            "cite_apa": str(row['cite_apa']).strip(),
            "reference_apa": str(row['reference_apa']).strip()
        }
    st.success("Metadatos de citas cargados exitosamente desde 'Fuentes Aula invertida.csv'.")
except Exception as e:
    st.warning(f"No se pudieron cargar los metadatos de citas desde 'Fuentes Aula invertida.csv': {e}. Las citas pueden no aparecer en el contenido generado. Asegúrate de que el archivo existe y el formato es correcto (delimitador ';').")

# --- Funciones Auxiliares ---
def nota_a_nivel(nota):
    if nota >= 85:
        return 0.85
    elif nota >= 55:
        return 0.55
    else:
        return 0.25

def actualizar_niveles_desde_csv_examen(estudiantes_df_local, uploaded_file):
    if uploaded_file is not None:
        try:
            exam_results_df = pd.read_csv(uploaded_file)
            st.subheader("Resultados del Examen Subido (Últimas Entradas):")
            st.dataframe(exam_results_df.tail(5))

            for index, row in exam_results_df.iterrows():
                estudiante_id = row.get("ID de Estudiante", None)
                nota_examen_general = row.get("Puntuación total", None)

                if pd.notna(estudiante_id) and pd.notna(nota_examen_general):
                    estudiante_id = int(estudiante_id)
                    nota_examen_general = int(nota_examen_general)

                    nuevo_nivel_general = nota_a_nivel(nota_examen_general)
                    idx = estudiantes_df_local[estudiantes_df_local["id"] == estudiante_id].index
                    if not idx.empty:
                        for concepto in conceptos:
                            estudiantes_df_local.loc[idx, concepto] = nuevo_nivel_general
                        st.write(
                            f"Nivel de estudiante **{estudiante_id}** actualizado a **{nivel_map[nuevo_nivel_general]}** basado en examen."
                        )
                    else:
                        st.warning(
                            f"Estudiante con ID **{estudiante_id}** del examen no encontrado en la base de datos local."
                        )
                else:
                    st.warning(
                        f"Fila {index+1} del examen subido: Faltan 'ID de Estudiante' o 'Puntuación total'."
                    )

            estudiantes_df_local.to_csv("estudiantes.csv", index=False)
            st.success(
                "Niveles de conocimiento actualizados desde el examen subido. Por favor, recarga la aplicación si deseas ver los cambios de inmediato."
            )
            return estudiantes_df_local
        except Exception as e:
            st.error(f"Error al procesar el archivo CSV de examen: {e}")
            st.info(
                "Asegúrate de que el archivo es un CSV válido y que las columnas 'ID de Estudiante' y 'Puntuación total' existen."
            )
    return estudiantes_df_local


# Función para generar embeddings (copiada del script de indexación)
@st.cache_data(show_spinner=False) # Caching para evitar regenerar el mismo embedding si la consulta se repite
def get_embedding(text, task_type="RETRIEVAL_QUERY", retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type=task_type # Usar RETRIEVAL_QUERY para consultas
            )
            return response['embedding']
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"Falló la generación de embedding después de {retries} intentos: {e}")
                return None


# Función principal para obtener contenido con RAG
# Se ha eliminado @st.cache_data para asegurar que la función siempre use la configuración más reciente del modelo
def obtener_contenido_gemini(estudiante_id, concepto_id, nivel_dificultad_texto):
    nombre_concepto = conceptos_legibles.get(concepto_id, concepto_id)
    # Define la "pregunta" que usaremos para buscar en Pinecone
    query_for_rag = f"Explica en detalle {nombre_concepto} para un nivel {nivel_dificultad_texto} en el contexto de Flipped Classroom."

    # 1. Generar el embedding de la pregunta
    query_embedding = get_embedding(query_for_rag, task_type="RETRIEVAL_QUERY")
    if query_embedding is None:
        return f"No se pudo generar el embedding para buscar información sobre {nombre_concepto}."

    # 2. Buscar en Pinecone para recuperar los fragmentos más relevantes
    context_chunks = []
    # relevant_citations se poblará ahora directamente de los metadatos de Pinecone para los fragmentos utilizados
    relevant_citations = set() # Usar un set para almacenar tuplas (cite_apa, reference_apa) y evitar duplicados

    with st.spinner(f"Buscando información relevante en la base de conocimiento para '{nombre_concepto}'..."):
        try:
            query_results = index.query(
                vector=query_embedding,
                top_k=5, # Recupera los 5 fragmentos más similares
                include_metadata=True # Asegúrate de obtener los metadatos que guardaste
            )

            for match in query_results.matches:
                chunk_text = match.metadata.get("original_text", "")
                
                # CRÍTICO: Obtener cite_apa y reference_apa DIRECTAMENTE de los metadatos de Pinecone.
                # Esto ASUME que estos campos fueron almacenados correctamente durante la indexación
                # (por ejemplo, en tu script de indexación, al hacer upsert, incluiste estos campos en 'metadata').
                cite_apa = match.metadata.get("cite_apa", "N/A_CITA_NO_DISPONIBLE") 
                reference_apa = match.metadata.get("reference_apa", "N/A_REFERENCIA_NO_DISPONIBLE")

                if chunk_text and cite_apa != "N/A_CITA_NO_DISPONIBLE":
                    # Incluir el texto del chunk Y su cita dentro del contexto que se le da a Gemini
                    context_chunks.append(f"FRAGMENTO: {chunk_text}\nCITA ASOCIADA: {cite_apa}\nREFERENCIA ASOCIADA: {reference_apa}")
                    relevant_citations.add((cite_apa, reference_apa))
                elif chunk_text:
                    # Si no hay una cita APA válida desde Pinecone, aún incluye el fragmento
                    # pero con una nota para que el modelo NO intente citarlo.
                    context_chunks.append(f"FRAGMENTO: {chunk_text}\nNOTA: Este fragmento no tiene una referencia APA completa disponible en el sistema para ser citado, o la referencia asociada no fue correctamente almacenada en Pinecone.")

        except Exception as e:
            st.error(f"Error al consultar Pinecone: {e}. Revisa la conexión y el índice.")
            return f"No se pudo recuperar información de la base de conocimiento para **{nombre_concepto}**."

    # Determinar los parámetros de generación según el nivel de dificultad
    word_count_target = 0
    depth_instruction = ""
    example_count_instruction = ""
    
    # Usuario solicitó específicamente 3 citas para todos los niveles
    citation_count_target = 3 

    if nivel_dificultad_texto == "Básico":
        word_count_target = 200
        depth_instruction = "Mantén la explicación sencilla, clara y directa, enfocándote en los fundamentos esenciales del concepto. Utiliza vocabulario fácil de entender."
        example_count_instruction = "1 ejemplo práctico."
    elif nivel_dificultad_texto == "Intermedio":
        word_count_target = 300
        depth_instruction = "Proporciona una explicación detallada, explorando conceptos clave y sus implicaciones. Introduce un nivel de detalle que facilite una comprensión más profunda sin abrumar."
        example_count_instruction = "2 ejemplos prácticos."
    elif nivel_dificultad_texto == "Avanzado":
        word_count_target = 380
        depth_instruction = "Ofrece una explicación profunda y exhaustiva, analizando matices, posibles desafíos, diferentes perspectivas o consideraciones avanzadas. Sintetiza información de múltiples fuentes del contexto para construir un argumento sólido y bien fundamentado."
        example_count_instruction = "3 ejemplos prácticos."
    
    # Construir la lista de referencias disponibles para Gemini
    available_references_string = ""
    if relevant_citations:
        # Asegurarse de que las referencias disponibles sean las que tienen cite_apa y reference_apa válidos
        filtered_references = [(c, r) for c, r in relevant_citations if c != "N/A_CITA_NO_DISPONIBLE"]
        if filtered_references:
            sorted_refs = sorted(list(filtered_references), key=lambda x: x[0]) # Ordenar por cite_apa
            available_references_string += "\n\n**REFERENCIAS DISPONIBLES PARA CITAR (ÚSALAS Y SOLO ESTAS):**\n"
            for cite, ref in sorted_refs:
                available_references_string += f"- {ref} (Cita: {cite})\n"
        else:
            available_references_string += "\n\n**No hay referencias válidas disponibles en el contexto para citar.**\n"
    else:
        available_references_string += "\n\n**No se encontraron fragmentos relevantes con referencias válidas en la base de conocimiento.**\n"

    # Construir el prompt para Gemini con el contexto recuperado
    context_string = "\n\n".join(context_chunks) if context_chunks else "No se encontró información relevante en los documentos."

    rag_prompt = f"""
    Eres un tutor educativo experto en la metodología Flipped Classroom.
    El estudiante con ID {estudiante_id} requiere contenido sobre **{nombre_concepto}**.
    El nivel de conocimiento del estudiante es **{nivel_dificultad_texto}**.

    **INFORMACIÓN DE CONTEXTO RELEVANTE (de tus documentos):**
    {context_string}

    {available_references_string}

    **INSTRUCCIONES PARA CREAR EL CONTENIDO PERSONALIZADO:**
    1. Explica claramente el concepto de **{nombre_concepto}**, **utilizando y refiriéndose ÚNICAMENTE a la "INFORMACIÓN DE CONTEXTO RELEVANTE" proporcionada.**
    2. **Adapta la complejidad, el detalle y la extensión del lenguaje al nivel {nivel_dificultad_texto} del estudiante:**
        - {depth_instruction}
        - El contenido debe tener aproximadamente **{word_count_target} palabras.**
    3. Incluye al menos **{example_count_instruction}** o escenarios relevantes relacionados con el método Flipped Classroom, adecuados al nivel.
    4. Usa un tono informativo, pedagógico y motivador.
    5. Formatea la respuesta en Markdown con encabezados (##, ###), listas y negritas para mejorar la legibilidad.
    6. **SOLO CITA FUENTES QUE TENGAN LA ETIQUETA "CITA ASOCIADA:" EN LA "INFORMACIÓN DE CONTEXTO RELEVANTE" O QUE APAREZCAN EN LA SECCIÓN "REFERENCIAS DISPONIBLES PARA CITAR".** Utiliza el formato (Autor, Año) o (Autor1 y Autor2, Año).
    7. **CRÍTICO: BAJO NINGUNA CIRCUNSTANCIA INVENTES, MODIFIQUES O AÑADAS CITAS O REFERENCIAS QUE NO ESTÉN EXPLÍCITAMENTE Y EN SU TOTALIDAD DENTRO DE LA "INFORMACIÓN DE CONTEXTO RELEVANTE" O LA SECCIÓN "REFERENCIAS DISPONIBLES PARA CITAR".** Si un fragmento no tiene una "CITA ASOCIADA:" (o su referencia completa no está en "REFERENCIAS DISPONIBLES PARA CITAR"), **NO LO CITES**. Simplemente explica el contenido si es relevante, pero sin atribuirlo a una fuente que no existe en tu contexto.
    8. Es fundamental que cites exactamente **{citation_count_target}** fuentes diferentes (si la "INFORMACIÓN DE CONTEXTO RELEVANTE" y "REFERENCIAS DISPONIBLES PARA CITAR" lo permiten y son suficientes). Si no hay suficientes fuentes válidas, cita las que haya, pero **NUNCA inventes.**
    9. Al final de tu explicación, añade una sección de "Referencias Bibliográficas" y **LISTA ESTRICTAMENTE SÓLO LAS REFERENCIAS COMPLETAS QUE HAYAS CITADO EN EL TEXTO, TOMÁNDOLAS DIRECTAMENTE DE LA SECCIÓN "REFERENCIAS DISPONIBLES PARA CITAR".** No añadas otras referencias.

    **Ejemplo de formato de respuesta:**
    ## La Definición de Flipped Classroom (Nivel: Intermedio)

    El concepto de Flipped Classroom, o aula invertida, ... (según Masero-Moreno & Morant, 2024).

    ### Ejemplos Prácticos
    1. Un ejemplo de ejemplo ...
    2. Otro escenario de ejemplo...

    ---
    **Referencias Bibliográficas:**
    - Masero-Moreno, I. C., & Morant, G. A. (2024). Implementacion e influencia del modelo de clase invertida en el aprendizaje en linea de dos asignaturas universitarias. EDUCAR, 60(1), Article 1. https://doi.org/10.5565/rev/educar.1765
    """

    try:
        with st.spinner(f"Generando contenido con IA avanzada (RAG) para '{nombre_concepto}'..."):
            response = model.generate_content(rag_prompt)
            generated_content = response.text
            return generated_content
    except Exception as e:
        st.error(f"Error al generar contenido con Gemini (RAG) para '{nombre_concepto}': {e}. Esto podría deberse a un problema con la API o al contenido.")
        return f"No se pudo generar contenido dinámico basado en documentos para **{nombre_concepto}**. Por favor, inténtalo de nuevo más tarde."
# --- Interfaz Streamlit ---

st.set_page_config(layout="wide", page_title="Sistema de Aprendizaje Personalizado")
st.title("Sistema de Aprendizaje Personalizado - Flipped Classroom")

st.subheader("1. Actualizar Niveles con Resultados de Examen (CSV)")
st.info(
    "Descarga el CSV de respuestas de tu Google Form y súbelo aquí para actualizar los niveles de conocimiento de los estudiantes."
)
uploaded_file = st.file_uploader("Sube el archivo CSV de respuestas del examen", type=["csv"])
if uploaded_file is not None:
    # Asegúrate de que estudiantes_df se actualice globalmente o se maneje con st.session_state
    estudiantes_df = actualizar_niveles_desde_csv_examen(estudiantes_df.copy(), uploaded_file)
    # Considera una recarga suave o uso de session_state para reflejar los cambios inmediatamente
    # st.experimental_rerun() # Descomentar si quieres un recarga forzada

st.subheader("2. Consulta tu Contenido Personalizado")
estudiante_id = st.number_input("Ingresa tu ID de estudiante", min_value=1, step=1, key="student_id_input")

if estudiante_id:
    estudiante = estudiantes_df[estudiantes_df["id"] == estudiante_id]
    if not estudiante.empty:
        st.write(f"**Nombre del estudiante**: {estudiante['nombre'].iloc[0]}")

        st.subheader("Tu Nivel de Conocimiento Actual:")
        niveles_para_grafico = []
        for concepto in conceptos:
            nivel_num = estudiante[concepto].iloc[0]
            nivel_texto = nivel_map.get(nivel_num, "Desconocido")
            niveles_para_grafico.append(
                {"Concepto": concepto, "Nivel": nivel_texto, "Valor": nivel_num}
            )

        niveles_df_chart = pd.DataFrame(niveles_para_grafico)

        chart = (
            alt.Chart(niveles_df_chart)
            .mark_bar()
            .encode(
                x=alt.X("Concepto:N", title="Concepto", sort=conceptos),
                y=alt.Y("Valor:Q", title="Nivel de Conocimiento", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "Nivel:N",
                    scale=alt.Scale(
                        domain=["Básico", "Intermedio", "Avanzado"],
                        range=["#ff9999", "#66b3ff", "#99ff99"],
                    ),
                ),
                tooltip=["Concepto", "Nivel", "Valor"],
            )
            .properties(width="container", height=300, title="Niveles de Conocimiento por Concepto")
            .configure_title(dy=-10)
        )

        st.altair_chart(chart, use_container_width=True)
        st.markdown("---")

        st.subheader("Contenido Personalizado para Ti:")
        for concepto in conceptos:
            nivel_num = estudiante[concepto].iloc[0]
            nivel_dificultad_texto = nivel_map.get(nivel_num, "Básico")
            with st.expander(f"Contenido para {conceptos_legibles.get(concepto, concepto).capitalize()} (Nivel: {nivel_dificultad_texto})"):
                st.markdown(obtener_contenido_gemini(estudiante_id, concepto, nivel_dificultad_texto))
    else:
        st.error("ID de estudiante no encontrado. Por favor, verifica el ID.")

st.subheader("3. Actualizar Nivel de Conocimiento Manualmente (Administrador)")
with st.form("actualizar_conocimiento_manual"):
    estudiante_id_update = st.number_input(
        "ID del estudiante a actualizar", min_value=1, step=1, key="manual_update_id"
    )
    concepto_update = st.selectbox("Concepto a actualizar", conceptos, key="manual_update_concept")
    nuevo_nivel = st.selectbox(
        "Nuevo nivel", [0.25, 0.55, 0.85], format_func=lambda x: nivel_map[x], key="manual_update_level"
    )
    submit = st.form_submit_button("Actualizar Nivel")

    if submit:
        idx = estudiantes_df[estudiantes_df["id"] == estudiante_id_update].index
        if not idx.empty:
            estudiantes_df.loc[idx, concepto_update] = nuevo_nivel
            estudiantes_df.to_csv("estudiantes.csv", index=False)
            st.success(
                f"Nivel de conocimiento para **{concepto_update}** actualizado a **{nivel_map[nuevo_nivel]}** para el estudiante ID {estudiante_id_update}."
            )
            # st.experimental_rerun() # Descomentar si quieres recarga forzada para ver el gráfico actualizado
        else:
            st.error("ID de estudiante no encontrado. No se pudo actualizar el nivel.")
