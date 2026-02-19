import os
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm

#Configuración de credenciales, reemplazar API_KEY con la OpenAI Api Key
os.environ['OPENAI_API_KEY'] = "API_KEY"
client = OpenAI()

#Parámetros
MODELOS = ["gpt-4o-mini", "gpt-4o", "gpt-5.1"]
NUM_EJECUCIONES = 5
TEMPERATURA = 0.0

#Dataset estratificado N=100 palabras
PALABRAS_TEST = [
    "sol", "pan", "luz", "mar", "gato", "perro", "casa", "mesa", "silla", "rojo",
    "azul", "uno", "dos", "tres", "flor", "tren", "bar", "pez", "oso", "ave",
    "amigo", "tiempo", "fuego", "tierra", "cielo", "playa", "bosque", "ciudad",
    "campo", "nube", "lluvia", "nieve", "viento", "trueno", "rayo", "piedra",
    "papel", "tijera", "juego", "libro", "musica", "arte", "cine", "teatro",
    "importante", "desarrollo", "tecnologia", "estructura", "transporte",
    "universidad", "inteligencia", "artificial", "programacion", "algoritmo",
    "matematicas", "estadistica", "probabilidad", "informacion", "comunicacion",
    "electricidad", "magnetismo", "biotecnologia", "astronomia", "filosofia",
    "desafortunadamente", "inconstitucionalidad", "esternocleidomastoideo",
    "electroencefalografista", "anticonstitucionalmente", "otorrinolaringologo",
    "paralelepipedo", "responsabilidad", "internacionalizacion", "multidisciplinario",
    "cigüeña", "pingüino", "vergüenza", "bilingüe", "pedigüeño",
    "canción", "árbol", "fácil", "música", "pájaro", "cáliz",
    "mañana", "niño", "piña", "montaña", "sueño", "año",
    "mexico", "argentina", "venezuela", "colombia", "españa",
    "einstein", "newton", "tesla", "curie"
]

#Funciones de verdad base
def mover_primera_letra(p):
    return p[1:] + p[0] if p else ""

def invertir_cadena(p):
    return p[::-1]

def contar_vocales(p):
    return str(sum(1 for c in p.lower() if c in "aeiouáéíóúü"))

#Definición de tareas algorítmicas y prompts
TAREAS = {
    "mover_primera": {
        "func": mover_primera_letra,
        "desc": "Mueve la primera letra de la palabra al final.",
        "ejemplos": [("google", "oogleg"), ("python", "ythonp"), ("java", "avaj")]
    },
    "invertir": {
        "func": invertir_cadena,
        "desc": "Invierte el orden de los caracteres de la palabra.",
        "ejemplos": [("casa", "asac"), ("roma", "amor"), ("lupa", "apul")]
    },
    "contar_vocales": {
        "func": contar_vocales,
        "desc": "Cuenta cuántas vocales (a,e,i,o,u) tiene la palabra. Responde solo con el número.",
        "ejemplos": [("banana", "3"), ("coche", "2"), ("murcielago", "5")]
    }
}

def generar_prompt(desc, input_word, ejemplos):
    prompt = f"Instrucción: {desc}\n\n"
    for inp, out in ejemplos:
        prompt += f"Entrada: {inp}\nSalida: {out}\n\n"
    prompt += f"Entrada: {input_word}\nSalida:"
    return prompt

#Función de inferencia LLM
def consultar_llm(modelo, prompt, estrategia):
    params = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Manejo dinámico del esfuerzo de razonamiento para la serie GPT-5.1/o1
    if "gpt-5" in modelo or "o1" in modelo:
        if estrategia == "Zero-Shot":
            params["reasoning_effort"] = "medium"
        else:
            params["reasoning_effort"] = "none"
            params["temperature"] = TEMPERATURA 
    else:
        params["temperature"] = TEMPERATURA

    try:
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

#Ejecución principal
def ejecutar_experimento():
    resultados = []
    estrategias = ["Zero-Shot", "One-Shot", "Few-Shot"]
    
    #Barra de progreso
    total_iteraciones = len(MODELOS) * len(TAREAS) * len(estrategias) * NUM_EJECUCIONES * len(PALABRAS_TEST)
    pbar = tqdm(total=total_iteraciones, desc="Ejecutando Inferencia")

    for modelo in MODELOS:
        for nombre_tarea, datos_tarea in TAREAS.items():
            for estrategia in estrategias:
                #Determinación del contexto
                if estrategia == "Zero-Shot":
                    ejemplos_activos = []
                elif estrategia == "One-Shot":
                    ejemplos_activos = datos_tarea['ejemplos'][:1]
                else:
                    ejemplos_activos = datos_tarea['ejemplos']

                aciertos = 0
                total_intentos = NUM_EJECUCIONES * len(PALABRAS_TEST)

                for _ in range(NUM_EJECUCIONES):
                    for palabra in PALABRAS_TEST:
                        prompt = generar_prompt(datos_tarea['desc'], palabra, ejemplos_activos)
                        respuesta_llm = consultar_llm(modelo, prompt, estrategia)

                        #Evaluación determinista
                        if not respuesta_llm.startswith("ERROR"):
                            respuesta_limpia = respuesta_llm.lower().replace("salida:", "").strip().rstrip(".")
                            verdad = datos_tarea['func'](palabra).lower()
                            
                            if respuesta_limpia == verdad:
                                aciertos += 1

                        pbar.update(1)

                #Registro de la precisión final
                precision = aciertos / total_intentos
                resultados.append({
                    "Modelo": modelo,
                    "Algoritmo": nombre_tarea,
                    "Estrategia": estrategia,
                    "Precisión": precision
                })

    pbar.close()
    return pd.DataFrame(resultados)

#Visualización de los resultados
def graficar_resultados(df):
    algoritmos = df['Algoritmo'].unique()
    fig, axes = plt.subplots(1, len(algoritmos), figsize=(18, 6), sharey=True)
    colores = {'Zero-Shot': '#4A90E2', 'One-Shot': '#F5A623', 'Few-Shot': '#7ED321'}

    for i, algo in enumerate(algoritmos):
        ax = axes[i] if len(algoritmos) > 1 else axes
        datos = df[df['Algoritmo'] == algo]
        pivot = datos.pivot(index='Modelo', columns='Estrategia', values='Precisión')
        pivot = pivot.reindex(columns=['Zero-Shot', 'One-Shot', 'Few-Shot'])

        pivot.plot(kind='bar', ax=ax, color=[colores[c] for c in pivot.columns], width=0.8, edgecolor='black')

        ax.set_title(f"Tarea: {algo.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=0)
        
        if i == 0:
            ax.set_ylabel("Precisión", fontsize=12)
        
        ax.legend(title="Contexto", loc='lower right')

    plt.suptitle(f"Rendimiento en Inducción de Algoritmos (N={len(PALABRAS_TEST)}, k={NUM_EJECUCIONES})", fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig("resultados_experimento_IIT.png", dpi=300)
    print("Gráfica guardada exitosamente'")

#Ejecución principal
if __name__ == "__main__":
    print("Iniciando validación experimental...")
    df_resultados = ejecutar_experimento()
    print("\nResultados:")
    print(df_resultados)
    graficar_resultados(df_resultados)