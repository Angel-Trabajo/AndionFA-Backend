# En tu aplicación Python principal:
from mic_simple_system import MICCluster

# Crear clúster
cluster = MICCluster()

# Ejecutar trabajo pesado
def procesar_con_mic(datos_pesados):
    # Convertir datos a formato para MIC
    tamaño_problema = len(datos_pesados)
    
    # Ejecutar en las 4 MIC
    resultados = cluster.run_parallel(
        total_size=tamaño_problema,
        strategy="equal"
    )
    
    # Procesar resultados
    return combinar_resultados(resultados)

# Usar en tu flujo
datos = cargar_datos_intensivos()
resultado = procesar_con_mic(datos)