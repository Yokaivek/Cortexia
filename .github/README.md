# 🧠 Cortexia - Sistema de Clasificación de Tumores Cerebrales

## Descripción del Proyecto

**Cortexia** es un sistema inteligente de apoyo diagnóstico que utiliza Inteligencia Artificial para clasificar tumores cerebrales a partir de imágenes de Resonancia Magnética (MRI) T1. Combina un modelo de aprendizaje profundo con un asistente clínico basado en IA para proporcionar recomendaciones médicas fundamentadas.

### Objetivo Principal
Asistir a profesionales médicos especializados en radiología y neurología en la interpretación de imágenes MRI y la toma de decisiones clínicas, proporcionando clasificaciones rápidas y explicaciones clínicas basadas en resultados de modelos de machine learning.

## 🎯 Características Principales

### 1. **Clasificador CNN Entrenado (Precisión: 89.6%)**
- Clasifica tumores cerebrales en 4 categorías:
  - **Glioma**: Tumor del sistema nervioso central
  - **Meningioma**: Tumor de las membranas cerebrales
  - **Tumor Pituitario**: Tumor de la glándula pituitaria
  - **Sin Tumor**: Ausencia de patología tumoral

### 2. **Pipeline de Limpieza de Datos Robusto**
- Detección y eliminación de imágenes duplicadas usando hash perceptual
- Separación limpia entre conjuntos de entrenamiento, validación y prueba
- Prevención de data leakage mediante validación cruzada estructurada

### 3. **Interfaz Web Intuitiva (Streamlit)**
- Carga simple de imágenes MRI
- Visualización clara de resultados del modelo
- Interfaz responsive (diseño claro y oscuro)

### 4. **Asistente Clínico IA (GPT-4o-mini)**
- Interpretación contextualizada de resultados
- Recomendaciones de estudios complementarios
- Chat interactivo para consultas adicionales
- Guardrails de seguridad estrictos para mantener el enfoque médico

## 📋 Requisitos del Sistema

### Dependencias
Instala los paquetes necesarios usando:
```bash
pip install -r requirements.txt
```

### Configuración de API
1. Obtén una clave API de OpenAI (https://platform.openai.com/api-keys)
2. Crea un archivo `.env.k` en la raíz del proyecto:
   ```
   OPENAI_API_KEY=tu_clave_api_aqui
   ```

## 🚀 Cómo Usar

### Ejecutar la Aplicación Web
```bash
streamlit run Despliegue.py
```
La aplicación se abrirá en `http://localhost:8501`

**Flujo de uso:**
1. Carga una imagen MRI en formato JPG o PNG
2. Haz clic en "Realizar diagnóstico"
3. Revisa los resultados del modelo (probabilidades por clase)
4. Consulta con el asistente clínico si necesitas más detalles

### Entrenar o Evaluar el Modelo
Abre `Proyecto.ipynb` en Jupyter/JupyterLab:
```bash
jupyter notebook Proyecto.ipynb
```

**Secciones principales del notebook:**
- **Celdas 1-2**: Configuración de GPU para TensorFlow
- **Celdas 3-8**: Análisis y limpieza del dataset (deduplicación)
- **Celdas 9-12**: Carga y preprocesamiento de imágenes
- **Celdas 13-18**: Definición y entrenamiento de la red CNN
- **Celdas 19-24**: Evaluación, matriz de confusión y pruebas de confianza

## 📁 Estructura del Proyecto

```
Proyecto 5/
├── Despliegue.py                    # Aplicación Streamlit principal
├── Proyecto.ipynb                   # Notebook de entrenamiento y análisis
├── prompts.py                       # Sistema de prompts para la IA clínica
├── requirements.txt                 # Dependencias del proyecto
├── .env.k                          # Archivo de configuración (API keys)
├── models_2/                        # Modelos entrenados (.keras)
│   ├── model_0.896.keras           # Mejor modelo (89.6% de precisión)
│   └── [otros modelos...].keras
├── Epic and CSCR hospital Dataset/  # Conjunto de datos
│   ├── Train/                       # Datos de entrenamiento
│   ├── Train_clean/                 # Datos de entrenamiento limpios
│   ├── Test/                        # Datos de prueba original
│   ├── Test_clean/                  # Datos de prueba limpios
│   ├── Test_clean_plus/             # Datos de prueba sin duplicados
│   ├── Test_final/                  # Conjunto final de prueba
│   └── Validacion/                  # Conjunto de validación
└── .github/                         # Documentación (este archivo)
```

## 🔬 Detalles Técnicos

### Arquitectura de la Red
La CNN (Convolutional Neural Network) consta de:
- **4 bloques convolucionales** con BatchNormalization y Dropout progresivo
- **Activaciones LeakyReLU** (alpha=0.1) para mejor convergencia
- **Capas densas** con regularización (dropout 50%)
- **Función de activación final**: Softmax (4 clases)

```python
# Estructura simplificada:
Conv2D(32) → BN → LeakyReLU → MaxPool → Dropout(0.25)
Conv2D(64) → BN → LeakyReLU → MaxPool → Dropout(0.30)
Conv2D(128) → BN → LeakyReLU → MaxPool → Dropout(0.35)
Conv2D(256) → BN → LeakyReLU → MaxPool → Dropout(0.40)
Flatten → Dense(256) → BN → LeakyReLU → Dropout(0.5) → Dense(4, softmax)
```

### Preprocesamiento de Imágenes
- **Redimensionamiento**: 224×224 píxeles
- **Normalización ImageNet**: 
  - Media: [0.485, 0.456, 0.406]
  - Desviación estándar: [0.229, 0.224, 0.225]
- **Augmentación de datos** (durante entrenamiento):
  - Rotación: ±15°
  - Desplazamiento: ±20%
  - Zoom: ±25%
  - Volteo horizontal: Sí

### Deduplicación de Dataset
- **Método**: Hash perceptual (pHash) para detectar imágenes similares
- **Distancia umbral**: ≤1 (muy estricto para evitar data leakage)
- **Prevención de fuga**: Se comprueban imágenes entre conjuntos de entrenamiento y prueba

## 📊 Resultados y Evaluación

### Métricas del Modelo (Test_final)
- **Precisión global**: 89.6%
- **Matriz de confusión**: Disponible en `Proyecto.ipynb`
- **F1-Score por clase**: Reportado al final del notebook

### Formato de Predicción
La aplicación retorna un diccionario con:
```python
{
    "Predicción de clase": "Glioma",                    # Clase principal
    "Confianza": 0.856,                                  # 0.0-1.0
    "Probabilidad_no_tumor": 0.12,
    "Probabilidad_tumor": 0.88,
    "Subtipos tumorales": {                             # Sin "No Tumor"
        "Glioma": 0.856,
        "Meningioma": 0.089,
        "Pituitary": 0.035
    },
    "Todas las probabilidades": {                       # 4 clases
        "Glioma": 0.856,
        "Meningioma": 0.089,
        "No Tumor": 0.012,
        "Pituitary": 0.035
    }
}
```

## ⚠️ Disclaimers Importantes

**Esta herramienta es un ASISTENTE, no un diagnóstico definitivo:**

1. Los resultados del modelo son complementarios y deben ser interpretados por un radiólogo o neurólogo especializado
2. Siempre revisar las imágenes MRI directamente en caso de baja confianza (<75%)
3. Considerar el contexto clínico completo del paciente
4. No reemplaza la evaluación clínica profesional ni la experiencia médica

## 🛠️ Desarrollo y Mantenimiento

### Agregar una Nueva Clase de Tumor
1. Preparar dataset con imágenes en nuevas carpetas de clase
2. Actualizar `CLASS_NAMES` en `Despliegue.py` y `Proyecto.ipynb`
3. Reentrenar la CNN y guardar como `model_{accuracy:.3f}.keras`
4. Validar con matriz de confusión
5. Actualizar prompts en `prompts.py` con contexto clínico nuevo

### Actualizar Prompts Clínicos
- Editar `prompts.py` (variable `promp_fuerte`)
- Nunca modificar directivas de seguridad sin revisión médica
- Probar con casos ambiguos (baja confianza)

### Debugging Común
| Problema | Causa | Solución |
|----------|-------|----------|
| Modelo no encontrado | Extensión incorrecta | Verificar formato `.keras` en `models_2/` |
| Error de API | Clave inválida | Revisar `.env.k` y credenciales OpenAI |
| Carga de imagen falla | Formato no soportado | Solo JPG/PNG; revisar extensión |
| Baja confianza predicción | Dataset ambiguo | Revisar imagen + contexto clínico |

## 📦 Versiones y Compatibilidad

- **Python**: 3.8+
- **TensorFlow**: 2.14.0+
- **Streamlit**: 1.28.0+
- **CUDA/GPU**: Opcional (configurado con memory growth dinámico)

## 📚 Referencias Médicas

**Tipos de tumores cerebrales:**
- **Glioma**: Tumores de células gliales, incluyen astrocitomas y oligodendrogliomas
- **Meningioma**: Tumores benignos/malignos de las meninges (membranas cerebrales)
- **Tumor Pituitario**: Tumores de la glándula pituitaria (hipófisis)

**Especificación de imagen:**
- MRI T1 (potenciación T1)
- Incluye o excluye gadolinio según protocolo
- Resolución típica: 1×1mm a 3×3mm por píxel

## 🤝 Contribuciones y Mejoras

Áreas de futuro desarrollo:
- [ ] Agregar más clases tumorales (tumores secundarios, meningiomas atípicos)
- [ ] Integración con PACS (Sistema de Archivos y Comunicación de Imágenes)
- [ ] Exportación de reportes en PDF
- [ ] Modelo multi-modal (T1, T2, FLAIR)
- [ ] Validación externa en cohorte internacional

## 📧 Contacto y Soporte

Para preguntas o reportar problemas, contacta al equipo de desarrollo.

---

**Última actualización**: 15 de febrero de 2026  
**Versión del modelo**: 0.896 (89.6% de precisión)  
**Estado del proyecto**: En producción
