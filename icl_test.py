import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from tqdm import tqdm

# Configuración del entorno
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "TU_API_KEY_AQUI"))

# Parámetros experimentales
MODELOS = ["gpt-4.1-mini", "gpt-4.1", "gpt-5.1"]
NUM_EJECUCIONES = 5
TEMPERATURA = 0.0

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

def mover_primera_letra(p: str) -> str:
    """Traslada el carácter inicial al final de la cadena."""
    return p[1:] + p[0] if p else ""

def invertir_cadena(p: str) -> str:
    """Revierte el orden total de los caracteres."""
    return p[::-1]

def contar_vocales(p: str) -> str:
    """Devuelve la cantidad de vocales en la cadena como string."""
    return str(sum(1 for c in p.lower() if c in "aeiouáéíóúü"))

# Definición de tareas y ejemplos (In-Context Learning)
ALGORITMOS = {
    "mover_primera": {
        "func": mover_primera_letra,
        "desc": "Mueve la primera letra de la palabra al final.",
        "ejemplos": [("google", "oogleg"), ("python", "ythonp"), ("cielo", "ieloc")]
    },
    "invertir": {
        "func": invertir_cadena,
        "desc": "Invierte el orden de los caracteres de la palabra.",
        "ejemplos": [("casa", "asac"), ("roma", "amor"), ("luz", "zul")]
    },
    "contar_vocales": {
        "func": contar_vocales,
        "desc": "Cuenta cuántas vocales (a,e,i,o,u) tiene la palabra. Responde solo con el número.",
        "ejemplos": [("banana", "3"), ("coche", "2"), ("sol", "1")]
    }
}

def generar_prompt(instruccion: str, palabra: str, ejemplos: list) -> str:
    """Construye el prompt en formato estándar basado en la estrategia n-shot."""
    prompt = f"Instrucción: {instruccion}\n\n"
    for inp, out in ejemplos:
        prompt += f"Entrada: {inp}\nSalida: {out}\n\n"
    prompt += f"Entrada: {palabra}\nSalida: "
    return prompt

def consultar_llm(modelo: str, prompt: str, usa_razonamiento: bool = False) -> str:
    """Ejecuta la inferencia contra la API de OpenAI gestionando el modo de razonamiento."""
    params = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if 'gpt-5' in modelo:
        if usa_razonamiento:
            params['reasoning_effort'] = 'medium'
        else:
            params['temperature'] = TEMPERATURA
    else:
        params['temperature'] = TEMPERATURA

    try:
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "ERROR"

def ejecutar_evaluacion() -> pd.DataFrame:
    """Función principal que orquesta el experimento masivo."""
    resultados = []
    estrategias = ["Zero-Shot", "One-Shot", "Few-Shot"]

    for modelo in MODELOS:
        for nombre_algo, datos_algo in ALGORITMOS.items():
            for estrategia in estrategias:
                if estrategia == "Zero-Shot":
                    ejemplos = []
                elif estrategia == "One-Shot":
                    ejemplos = [datos_algo['ejemplos'][0]]
                else:
                    ejemplos = datos_algo['ejemplos']

                usa_razonamiento = ('gpt-5' in modelo)
                total_iters = len(PALABRAS_TEST) * NUM_EJECUCIONES
                pbar = tqdm(total=total_iters, desc=f"Evaluando: {modelo} | {nombre_algo} | {estrategia}", leave=False)

                for _ in range(NUM_EJECUCIONES):
                    for palabra in PALABRAS_TEST:
                        time.sleep(0.15) # Rate limiting
                        
                        prompt = generar_prompt(datos_algo['desc'], palabra, ejemplos)
                        respuesta_llm = consultar_llm(modelo, prompt, usa_razonamiento)

                        acierto = 0
                        if respuesta_llm and respuesta_llm != "ERROR":
                            respuesta_limpia = respuesta_llm.lower().replace("salida:", "").replace("result:", "").strip()
                            if respuesta_limpia.endswith("."):
                                respuesta_limpia = respuesta_limpia[:-1]

                            if respuesta_limpia == datos_algo['func'](palabra).lower():
                                acierto = 1

                        longitud_str = "Corta (<=5)" if len(palabra) <= 5 else "Larga (>5)"

                        resultados.append({
                            "Modelo": modelo,
                            "Algoritmo": nombre_algo,
                            "Estrategia": estrategia,
                            "Palabra": palabra,
                            "Longitud": longitud_str,
                            "Acierto": acierto
                        })
                        pbar.update(1)
                pbar.close()

    return pd.DataFrame(resultados)

def graficar_rendimiento(df: pd.DataFrame):
    """Genera y exporta la gráfica de rendimiento global."""
    df_global = df.groupby(['Modelo', 'Algoritmo', 'Estrategia'])['Acierto'].mean().reset_index()
    algos = df_global['Algoritmo'].unique()
    
    fig, axes = plt.subplots(1, len(algos), figsize=(20, 6), sharey=True)
    colores = {'Zero-Shot': '#FF6F61', 'One-Shot': '#6B5B95', 'Few-Shot': '#88B04B'}

    for i, algo in enumerate(algos):
        ax = axes[i] if len(algos) > 1 else axes
        datos_algo = df_global[df_global['Algoritmo'] == algo]
        pivot = datos_algo.pivot(index='Modelo', columns='Estrategia', values='Acierto')
        pivot = pivot.reindex(columns=['Zero-Shot', 'One-Shot', 'Few-Shot'])

        pivot.plot(kind='bar', ax=ax, color=[colores.get(x, '#333') for x in pivot.columns], width=0.8)

        ax.set_title(f"Tarea: {algo.upper()}", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xlabel("Modelo Evaluado", fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        
        if i == 0:
            ax.set_ylabel("Precisión", fontsize=12)
        
        ax.legend(title="Estrategia", loc='lower right')

    plt.suptitle("Rendimiento de Inducción Algorítmica", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.savefig("figura_1_rendimiento.png", dpi=300, bbox_inches='tight')
    plt.close()

def graficar_impacto_longitud(df: pd.DataFrame):
    """Genera y exporta la gráfica del impacto de longitud léxica."""
    df_filtrado = df[(df['Modelo'] == 'gpt-5.1') & (df['Algoritmo'] == 'invertir')]
    if df_filtrado.empty:
        return 
        
    pivot = df_filtrado.groupby(['Estrategia', 'Longitud'])['Acierto'].mean().unstack()
    pivot = pivot.reindex(['Zero-Shot', 'Few-Shot'])
    
    colores_longitud = {'Corta (<=5)': '#4E79A7', 'Larga (>5)': '#E15759'}
    
    ax = pivot.plot(kind='bar', figsize=(8, 6), color=[colores_longitud.get(x, '#333') for x in pivot.columns])
    
    plt.title("Impacto Léxico en Tarea de Inversión (GPT-5.1)", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.05)
    plt.ylabel("Precisión", fontsize=12)
    plt.xlabel("Estrategia", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Longitud", loc='lower left')
    
    plt.tight_layout()
    plt.savefig("figura_2_longitud.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Iniciando evaluación experimental...")
    df_resultados = ejecutar_evaluacion()
    
    print("Generando visualizaciones...")
    graficar_rendimiento(df_resultados)
    graficar_impacto_longitud(df_resultados)
    
    df_resultados.to_csv("datos_experimentales.csv", index=False)
    print("Evaluación finalizada. Datos y figuras exportados correctamente.")
