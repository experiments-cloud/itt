# 📄 Material Suplementario: Aprendizaje en Contexto en Modelos Lingüísticos Grandes

Este repositorio contiene el código fuente y el conjunto de datos integrado para reproducir los experimentos detallados en el artículo **"Aprendizaje en Contexto en Modelos Lingüísticos Grandes: Un Análisis Computacional"**, sometido a la revista *Ingeniería Investigación y Tecnología*.

El script automatiza la evaluación del rendimiento de tres modelos generativos (`gpt-4o-mini`, `gpt-4o` y `gpt-5.1`) bajo tres paradigmas de contexto (*Zero-Shot*, *One-Shot* y *Few-Shot*), aplicando tareas de inducción algorítmica y manipulación de cadenas con un enfoque determinista ($T=0$).

---

## 🛠️ Requisitos del Sistema

Para ejecutar el experimento, necesitas tener instalado **Python 3.8 o superior**. Las dependencias requeridas son mínimas y se centran en la inferencia y visualización de datos.

Puedes instalar las librerías necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install openai pandas matplotlib tqdm
