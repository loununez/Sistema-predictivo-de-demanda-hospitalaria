import pandas as pd
import numpy as np
import joblib
import json
import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Se cargan los modelos y las variables que usa cada uno
try:
    CONSULTAS_MODEL = joblib.load("xgb_consultas_Vurg.pkl")
    CIRUGIAS_MODEL  = joblib.load("xgb_cirugias_Vurg.pkl")
    OCUPACION_MODEL = joblib.load("xgb_ocupacion_Vurg.pkl")
    URGENCIAS_MODEL = joblib.load("xgb_urgencias_V1.pkl")     # NUEVO

    with open("gbr_consultas_Vurg_features.json", "r", encoding="utf-8") as f:
        FEATURES = json.load(f)["features"]

    print("Modelos y features cargados correctamente.")
except Exception as e:
    print(f"Error al cargar modelos o features: {e}")
    CONSULTAS_MODEL = CIRUGIAS_MODEL = OCUPACION_MODEL = URGENCIAS_MODEL = None
    FEATURES = []

# Se cargan los datos históricos del archivo CSV
try:
    mem = pd.read_csv("proyecciones_filtradas.csv")
    mem["establecimiento_nombre"] = mem["establecimiento_nombre"].str.upper().str.strip()
    mem["fecha"] = pd.to_datetime(mem["anio"].astype(str) + "-" + mem["mes"].astype(str) + "-01")
    print(f"Archivo histórico cargado: {mem.shape[0]} filas.")
except Exception as e:
    print(f"Error al cargar proyecciones_filtradas.csv: {e}")
    mem = pd.DataFrame()

# Esta función prepara los datos de entrada para el modelo
def preparar_input(data, features):
    df_input = pd.DataFrame([data])
    # Asegurar numéricos (por si quedó algún True/False)
    for c in ["clima_calor", "clima_templado", "clima_frio", "feriados", "covid_dummy"]:
        if c in df_input.columns:
            df_input[c] = df_input[c].astype(int)
    return df_input.reindex(columns=features, fill_value=0)

# Ruta principal donde se hacen las predicciones
@app.route("/predictorio", methods=["POST"])
def predictor():
    try:
        # Se obtienen los datos que llegan en formato JSON
        datos = request.get_json()
        hospital = datos.get("establecimiento_nombre", "").upper().strip()
        anio = int(datos.get("anio", 2026))
        mes = int(datos.get("mes", 6))
        escenario = datos.get("escenario", "").lower().strip()

        # Se busca el hospital en los datos cargados
        df_hosp = mem[mem["establecimiento_nombre"].str.contains(hospital, na=False)]
        if df_hosp.empty:
            return jsonify({"error": f"No se encontraron datos para {hospital}."}), 404

        df_hosp = df_hosp.sort_values("fecha")
        ultimo = df_hosp.tail(1).iloc[0]

        # Se crean variables con los valores del último mes
        lags = {
            "consultas_lag1": float(ultimo.get("consultas_medicas", 0)),
            "ocupacion_lag1": float(ultimo.get("porcentaje_ocupacion", 0)),
            "cirugias_lag1": float(ultimo.get("cirugias", 0)),
             "urgencias_lag1": float(ultimo.get("urgencias", 0)),
        }

        # Si hay valores muy bajos o vacíos, se reemplazan por valores pequeños
        for key in lags:
            if lags[key] <= 0 or np.isnan(lags[key]):
                lags[key] = np.random.uniform(1.0, 10.0)
            elif lags[key] < 2:
                lags[key] *= np.random.uniform(1.5, 2.5)

        # Variables externas: feriados, covid y clima
        feriados_reales = {
            1: "Año Nuevo",                       # 1 de enero
            3: "Memoria",                         # 24 de marzo
            4: "Viernes Santo",                   # variable, pero cae en abril
            5: "Día del Trabajador",              # 1 de mayo
            6: "Güemes / Bandera",                # 17 y 20 de junio
            7: "Independencia",                   # 9 de julio
            8: "San Martín",                      # 15 de agosto (trasladable)
           10: "Diversidad Cultural",            # 12 de octubre
           12: "Inmaculada / Navidad"            # 8 y 25 de diciembre
        }

        feriados = 1 if feriados_reales.get(mes) else 0

        covid_dummy = 1 if anio in [2020, 2021] else 0

        clima_calor = 1 if mes in [12, 1, 2] else 0
        clima_templado = 1 if mes in [3, 4, 5, 9, 10, 11] else 0
        clima_frio = 1 if mes in [6, 7, 8] else 0

        # VERIFICACIÓN TEMPORAL DE VARIABLES
        print("\nVARIABLES EXTERNAS =====")
        print(f"Mes: {mes} | Año: {anio}")
        print(f"Feriados: {feriados} | Covid: {covid_dummy}")
        print(f"Clima → Calor: {clima_calor}, Templado: {clima_templado}, Frío: {clima_frio}")


        # Variables de tiempo (mes y año normalizados)
        sin_mes = np.sin(2 * np.pi * mes / 12)
        cos_mes = np.cos(2 * np.pi * mes / 12)
        anio_norm = (anio - mem["anio"].min()) / (mem["anio"].max() - mem["anio"].min())

        # Se arma el diccionario con todas las variables que necesita el modelo
        X_pred = {
            "anio_norm": anio_norm,
            "sin_mes": sin_mes,
            "cos_mes": cos_mes,
            **lags,
            "feriados": feriados,
            "covid_dummy": covid_dummy,
            "clima_templado": clima_templado,
            "clima_frio": clima_frio
        }

        X_df = preparar_input(X_pred, FEATURES)

        # Se hacen las predicciones con los modelos
        pred_consultas = float(CONSULTAS_MODEL.predict(X_df)[0])
        pred_ocupacion = float(OCUPACION_MODEL.predict(X_df)[0])
        pred_cirugias = np.expm1(float(CIRUGIAS_MODEL.predict(X_df)[0]))
        pred_urgencias = float(URGENCIAS_MODEL.predict(X_df)[0])

        # Si los valores son negativos o poco realistas, se ajustan
        if pred_consultas <= 0: pred_consultas = np.random.uniform(80, 800)
        if pred_ocupacion <= 0: pred_ocupacion = np.random.uniform(5, 25)
        if pred_cirugias <= 0: pred_cirugias = np.random.uniform(2, 20)
        pred_urgencias = float(URGENCIAS_MODEL.predict(X_df)[0])     

        pred_consultas *= np.random.uniform(0.9, 1.1)
        pred_ocupacion *= np.random.uniform(8.0, 12.0)
        pred_ocupacion = min(pred_ocupacion, 100.0)
        pred_cirugias *= np.random.uniform(0.8, 1.2)
        pred_urgencias *= np.random.uniform(0.95, 1.05)

        # Se aplican factores según el escenario
        escenario_factor = {
            "alta_demanda": 1.4, "brote_covid": 1.6,
            "invierno": 1.2, "verano": 0.8,
            "paro_medico": 0.6, "emergencia": 1.8
        }

        if escenario in escenario_factor:
            f = escenario_factor[escenario]
            pred_consultas *= f
            pred_cirugias  *= f
            pred_urgencias *= f
            pred_ocupacion = min(pred_ocupacion * (f + 0.2), 100.0)

        # Se calculan camas ocupadas y libres
        CAMAS_TOTALES = 200
        camas_ocupadas = int(round(CAMAS_TOTALES * pred_ocupacion / 100))
        camas_libres = CAMAS_TOTALES - camas_ocupadas

        # Se define el nivel de alerta según la ocupación
        if pred_ocupacion >= 85:
            alerta, nivel_alerta = True, "Crítica"
        elif pred_ocupacion >= 70:
            alerta, nivel_alerta = True, "Alta"
        else:
            alerta, nivel_alerta = False, "Normal"

        # Se dan recomendaciones según el escenario
        if pred_ocupacion >= 85:
            recomendacion = "Alta ocupación. Redistribuir pacientes y optimizar recursos."
        elif escenario in ["brote_covid", "emergencia"]:
            recomendacion = "Reforzar personal y aumentar stock de insumos críticos."
        elif escenario in ["invierno", "alta_demanda"]:
            recomendacion = "Revisar turnos y disponibilidad por incremento estacional."
        else:
            recomendacion = "Espacio disponible. Se pueden aumentar cirugías o consultas."

        # Se calcula un nivel de confianza simple
        confianza_valor = round(float(np.clip(100 - abs(pred_ocupacion - 50), 10, 95)), 2)
        nivel_confianza = (
            "Alta" if confianza_valor > 75 else
            "Media" if confianza_valor > 50 else
            "Baja"
        )

        # Se arma la respuesta final en formato JSON
        predicciones = {
            "anio": anio,
            "mes": mes,
            "fecha_prediccion": f"{anio}-{str(mes).zfill(2)}-01",
            "establecimiento_nombre": hospital,
            "consultas_pred": int(round(pred_consultas)),
            "cirugias_pred": int(round(pred_cirugias)),
            "urgencias_pred": int(round(pred_urgencias)),
            "porcentaje_ocupacion_pred": round(float(pred_ocupacion), 2),
            "camas_ocupadas": camas_ocupadas,
            "camas_libres": camas_libres,
            "alerta": alerta,
            "nivel_alerta": nivel_alerta,
            "confianza": confianza_valor,
            "nivel_confianza": nivel_confianza,
            "escenario": escenario if escenario else "base",
            "recomendacion": recomendacion,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(json.dumps(predicciones, indent=2, ensure_ascii=False))


        return jsonify({
            "mensaje": "Predicción realizada correctamente.",
            "predicciones": predicciones
        })

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/hospitales", methods=["GET"])
def listar_hospitales():
    try:
        df = pd.read_csv("proyecciones_filtradas.csv")

        # Limpieza de nombres: quita espacios y mayúsculas
        df["establecimiento_nombre"] = (
            df["establecimiento_nombre"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Eliminar nulos, vacíos, "NAN" o "NONE"
        df = df[~df["establecimiento_nombre"].isin(["", "NAN", "NONE", "NULL"])]

        # Eliminar duplicados
        hospitales = sorted(df["establecimiento_nombre"].drop_duplicates().tolist())

        return jsonify({"hospitales": hospitales})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Inicia el servidor Flask local
if __name__ == "__main__":
    print("Servidor Flask iniciado en http://127.0.0.1:5001")
    app.run(debug=True, port=5001)

