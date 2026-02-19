# 📄 Material Suplementario: Aprendizaje en Contexto en Modelos Lingüísticos Grandes

Este repositorio contiene el código fuente y el conjunto de datos integrado para reproducir los experimentos detallados en el artículo **"Aprendizaje en Contexto en Modelos Lingüísticos Grandes: Un Análisis Computacional"**, sometido a la revista *Ingeniería Investigación y Tecnología*.

El script automatiza la evaluación del rendimiento de tres modelos generativos (`gpt-4o-mini`, `gpt-4o` y `gpt-5.1`) bajo tres paradigmas de contexto (*Zero-Shot*, *One-Shot* y *Few-Shot*), aplicando tareas de inducción algorítmica y manipulación de cadenas con un enfoque determinista ($T=0$).

---

## 🛠️ Requisitos del Sistema

Para ejecutar el experimento, necesitas tener instalado **Python 3.8 o superior**. Las dependencias requeridas son mínimas y se centran en la inferencia y visualización de datos.

Puedes instalar las librerías necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install openai pandas matplotlib tqdm
```

## 🚀 Instrucciones de Ejecución

**1. Configuración de la API Key**

El experimento hace llamadas a la API oficial de OpenAI. Por motivos de seguridad, la clave de acceso no está incluida en el código de este repositorio.

Antes de ejecutar el script, abre el archivo experimento2.py y reemplaza la cadena "TU_API_KEY_AQUI" (Línea 8) con tu clave secreta de OpenAI:
Python

```python
# ocl_test.py
os.environ['OPENAI_API_KEY'] = "sk-proj-tu_clave_real_aqui..."
```

**2. Ejecutar el script**

Ejecuta el experimento desde tu terminal:

```bash
python icl_test.py
```

**3. Restricciones operativas**

El script realiza miles de llamadas a la API para asegurar la validez estadística del muestreo. Dependiendo de los límites de tu nivel de facturación en OpenAI (Requests Per Minute / Tokens Per Day), la ejecución completa tomará varios minutos.
El script incluye una barra de progreso (tqdm) que mostrará el avance general y el tiempo estimado de finalización.

## 📊 Resultados 

Al concluir, el script generará de forma automática:

    Datos tabulares (Consola): Un DataFrame de Pandas con el cálculo de precisión (0.0−1.0) de las métricas obtenidas por cada modelo bajo cada paradigma de contexto.

    Gráfica de Alta Resolución (Archivo): Se exportará el archivo resultados_experimento_IIT.png en el mismo directorio. 

## 📂 Estructura del Repositorio

```Plaintext
/
├── icl_test.py                    # Script principal
├── LICENSE                        # Licencia de uso
└── README.md                      # Documentación del repositorio
```

