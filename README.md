# Modelo Predictivo de Deterioro de Salud en Pacientes Hospitalizados  

**Grupo n°76 – Vertical Data Science (HealthTech)**  
**Proyecto desarrollado en No Country**  

---

## Descripción General

Este proyecto tiene como propósito **anticipar la demanda hospitalaria** en los establecimientos de salud de la Provincia de Buenos Aires mediante técnicas de *machine learning*.  
A partir del análisis de datos históricos, el sistema **predice la evolución mensual** de:

- Consultas médicas  
- Cirugías  
- Urgencias  
- Porcentaje de ocupación hospitalaria  

El modelo busca **mejorar la planificación y gestión hospitalaria**, ayudando a anticipar picos de demanda, optimizar la disponibilidad de camas y personal, y prevenir situaciones de saturación.

---

## Objetivos del Proyecto

- Analizar tendencias históricas (2005–2023) de rendimiento hospitalario.  
- Entrenar modelos predictivos basados en *machine learning* (Prophet, XGBoost).  
- Desarrollar una **API REST** para exponer las predicciones de manera dinámica.  
- Conectar la API a un **dashboard interactivo en Power BI** que muestre la evolución de la demanda entre 2023 y 2026.  

---

## Arquitectura del Sistema


- Dataset original (Ministerio de Salud PBA)
- Procesamiento y limpieza de datos (Python / Pandas)
- Entrenamiento de modelos (XGBoost)
- Generación de proyecciones mensuales (2024–2026)
- API Flask para servir las predicciones en formato JSON
- Dashboard en Power BI (visualización interactiva)


## Fuente de Datos

Datos públicos del Ministerio de Salud de la Provincia de Buenos Aires, disponibles en el portal de datos abiertos:
🔗 Rendimientos de Establecimientos de Salud

El dataset contiene información sobre:

- Ocupación de camas
- Consultas médicas
- Cirugías
- Urgencias
- Personal y servicios
- Variables temporales

## Tecnologías Utilizadas

| Componente              | Tecnología       |
| ----------------------- | ---------------- |
| Lenguaje principal      | Python 3.10      |
| Modelado predictivo     | XGBoost          |
| Procesamiento de datos  | Pandas, NumPy    |
| API REST                | Flask            |
| Visualización           | Power BI         |
| Almacenamiento temporal | CSV / JSON       |


## Modelos Implementados

| Variable          | Modelo            | R²   | Descripción                           |
| ----------------- | ----------------- | ---- | ------------------------------------- |
| Consultas médicas | XGBoost Regressor | 0.96 | Precisión alta en patrones temporales |
| Cirugías          | XGBoost (log)     | 0.94 | Estacionalidad controlada             |
| Urgencias         | XGBoost           | 0.93 | Alta estabilidad ante variabilidad    |
| Ocupación (%)     | XGBoost           | 0.90 | Ajuste robusto ante valores extremos  |


## API REST

/predictorio → POST

Devuelve la proyección para un hospital y mes determinados.

Ejemplo de solicitud:

<img width="704" height="615" alt="image" src="https://github.com/user-attachments/assets/accf1990-f17c-4b26-85cd-5269e4d26ae3" />



Ejemplo de respuesta:

<img width="715" height="729" alt="image" src="https://github.com/user-attachments/assets/e7856c59-db4d-4412-a452-9dfccfb9e3db" />


## Dashboard Power BI

El dashboard interactivo muestra la evolución proyectada y el estado actual del sistema hospitalario.
Conecta directamente a la API Flask y actualiza automáticamente las predicciones.

Páginas principales:

Visión general: KPIs de consultas, cirugías, urgencias y ocupación.

Evolución temporal: análisis de tendencias y estacionalidad (2024–2026).

Detalle por hospital: nivel de alerta, recomendaciones y confianza del modelo.

## Equipo de Desarrollo

Grupo n°76 – Vertical Data Science / HealthTech
- Facundo Sardo
- Ramón Ramírez
- Gastón Peló
- Belén Urbaneja
- Lourdes Núñez
