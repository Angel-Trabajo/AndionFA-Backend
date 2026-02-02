"""
TU APLICACIÓN CON ACELERACIÓN POR 4 INTEL XEON PHI
"""

import numpy as np
from mic_simple_system import MICCluster
import json
from datetime import datetime

class MiAppAcelerada:
    def __init__(self):
        self.cluster = MICCluster()
        self.cache_resultados = {}
    
    def preprocesar_datos(self, datos):
        """Prepara datos para procesamiento en MIC"""
        # Convertir a formato optimizado
        return {
            'tamaño': len(datos),
            'datos_brutos': datos,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'tipo': 'procesamiento_intensivo'
            }
        }
    
    def ejecutar_en_mic(self, datos_preprocesados, tamaño_lote=1000000):
        """Ejecuta el cómputo intensivo en las 4 MIC"""
        tamaño_total = datos_preprocesados['tamaño']
        
        print(f"Iniciando procesamiento en 4 MIC...")
        print(f"  Elementos totales: {tamaño_total:,}")
        print(f"  Tamaño por lote: {tamaño_lote:,}")
        print(f"  Núcleos disponibles: 240")
        
        # Dividir en lotes si es muy grande
        if tamaño_total > tamaño_lote * 10:
            lotes = tamaño_total // tamaño_lote
            print(f"  Dividiendo en {lotes} lotes...")
            
            resultados_totales = []
            for i in range(0, tamaño_total, tamaño_lote):
                lote_size = min(tamaño_lote, tamaño_total - i)
                print(f"  Procesando lote {i//tamaño_lote + 1}/{lotes}...")
                
                resultados = self.cluster.run_parallel(
                    total_size=lote_size,
                    strategy="equal"
                )
                resultados_totales.extend(resultados)
                
            return resultados_totales
        else:
            # Ejecutar todo de una vez
            return self.cluster.run_parallel(
                total_size=tamaño_total,
                strategy="equal"
            )
    
    def postprocesar_resultados(self, resultados_mic):
        """Combina y procesa resultados de las MIC"""
        print(f"\nCombinando resultados de {len(resultados_mic)} ejecuciones...")
        
        # Filtrar éxitos
        exitosos = [r for r in resultados_mic if r.get('success', False)]
        
        if not exitosos:
            raise Exception("Todas las ejecuciones en MIC fallaron")
        
        # Combinar resultados (depende de tu aplicación)
        resultado_final = {
            'timestamp': datetime.now().isoformat(),
            'mics_exitosas': len(exitosos),
            'tiempo_total': max(r.get('execution_time', 0) for r in exitosos),
            'elementos_procesados': sum(r.get('problem_size', 0) for r in exitosos),
            'rendimiento_total': sum(r.get('perf', 0) for r in exitosos),
            'resultados_individuales': exitosos
        }
        
        # Guardar para análisis
        archivo_resultados = f"resultados_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(archivo_resultados, 'w') as f:
            json.dump(resultado_final, f, indent=2)
        
        print(f"Resultados guardados en: {archivo_resultados}")
        return resultado_final
    
    def pipeline_completo(self, datos_entrada):
        """Pipeline completo de procesamiento"""
        print("="*70)
        print("INICIANDO PROCESAMIENTO ACELERADO POR INTEL XEON PHI")
        print("="*70)
        
        # 1. Preprocesamiento
        datos_prep = self.preprocesar_datos(datos_entrada)
        print(f"✓ Datos preparados: {datos_prep['tamaño']:,} elementos")
        
        # 2. Ejecución en MIC
        inicio = datetime.now()
        resultados_mic = self.ejecutar_en_mic(datos_prep)
        tiempo_ejecucion = (datetime.now() - inicio).total_seconds()
        
        print(f"✓ Ejecución en MIC completada en {tiempo_ejecucion:.2f} segundos")
        
        # 3. Postprocesamiento
        resultado_final = self.postprocesar_resultados(resultados_mic)
        
        print(f"\n{'='*70}")
        print(f"PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"{'='*70}")
        print(f"Resumen:")
        print(f"  Tiempo total: {resultado_final['tiempo_total']:.2f}s")
        print(f"  Elementos: {resultado_final['elementos_procesados']:,}")
        print(f"  Rendimiento: {resultado_final['rendimiento_total']:,.0f} elem/s")
        print(f"  MICs utilizadas: {resultado_final['mics_exitosas']}/4")
        
        return resultado_final

# Ejemplo de uso
if __name__ == "__main__":
    # Crear aplicación
    app = MiAppAcelerada()
    
    # Datos de ejemplo (reemplaza con tus datos reales)
    datos_ejemplo = np.random.rand(5000000)  # 5 millones de elementos
    
    # Ejecutar pipeline
    resultado = app.pipeline_completo(datos_ejemplo)
    
    print(f"\n✅ Procesamiento completado. Ver archivo JSON para resultados detallados.")