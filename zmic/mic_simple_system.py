"""
SISTEMA SIMPLIFICADO PARA 4 INTEL XEON PHI
Versión probada y corregida
"""

import subprocess
import concurrent.futures
import time
import json
from datetime import datetime

class MICCluster:
    def __init__(self):
        self.mics = [0, 1, 2, 3]
        self.executable = "mic_production.mic"
    
    def run_on_mic(self, mic_id, problem_size):
        """Ejecuta trabajo en una MIC específica"""
        print(f"[MIC {mic_id}] Iniciando {problem_size:,} elementos...")
        
        cmd = [
            "micnativeloadex",
            self.executable,
            "-d", str(mic_id),
            "-a", str(problem_size),
            "-v"
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            exec_time = time.time() - start_time
            
            # Parsear resultado
            output = result.stdout
            success = result.returncode == 0
            
            # Extraer métricas
            metrics = {}
            for line in output.split('\n'):
                if 'Execution time:' in line:
                    try:
                        metrics['time'] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif 'Performance:' in line:
                    try:
                        perf = line.split(':')[1].strip().split()[0]
                        metrics['perf'] = float(perf.replace(',', ''))
                    except:
                        pass
                elif 'Final sum:' in line:
                    try:
                        metrics['result'] = float(line.split(':')[1].strip())
                    except:
                        pass
            
            if success:
                print(f"  ✓ MIC {mic_id}: {exec_time:.2f}s, {metrics.get('perf', 0):,.0f} elem/s")
            else:
                print(f"  ✗ MIC {mic_id}: Error {result.returncode}")
            
            return {
                'mic_id': mic_id,
                'problem_size': problem_size,
                'success': success,
                'execution_time': exec_time,
                'output': output[-500:],
                **metrics
            }
            
        except Exception as e:
            print(f"  ✗ MIC {mic_id}: Error - {e}")
            return {
                'mic_id': mic_id,
                'problem_size': problem_size,
                'success': False,
                'error': str(e)
            }
    
    def run_parallel(self, total_size, strategy="equal"):
        """Ejecuta trabajo en las 4 MIC en paralelo"""
        print(f"\n{'='*60}")
        print(f"EJECUTANDO EN 4 MIC EN PARALELO")
        print(f"{'='*60}")
        print(f"Tamaño total: {total_size:,} elementos")
        print(f"Estrategia: {strategy}")
        print(f"MICs disponibles: {self.mics}")
        
        # Preparar tareas
        tasks = []
        if strategy == "equal":
            chunk = total_size // len(self.mics)
            for mic in self.mics:
                tasks.append((mic, chunk))
        elif strategy == "test":
            # Tamaños diferentes para prueba
            sizes = [1000000, 2000000, 3000000, 4000000]
            for i, mic in enumerate(self.mics):
                tasks.append((mic, sizes[i]))
        
        print(f"\nDistribución:")
        for mic, size in tasks:
            print(f"  MIC {mic}: {size:,} elementos")
        
        # Ejecutar en paralelo
        print(f"\nIniciando ejecución paralela...")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Enviar todas las tareas
            future_to_mic = {
                executor.submit(self.run_on_mic, mic, size): (mic, size)
                for mic, size in tasks
            }
            
            # Recolectar resultados
            for future in concurrent.futures.as_completed(future_to_mic):
                mic, size = future_to_mic[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error procesando MIC {mic}: {e}")
                    results.append({
                        'mic_id': mic,
                        'problem_size': size,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def print_summary(self, results):
        """Imprime resumen de ejecución"""
        successful = [r for r in results if r.get('success', False)]
        
        print(f"\n{'='*60}")
        print(f"RESUMEN DE EJECUCIÓN")
        print(f"{'='*60}")
        
        print(f"MICs exitosas: {len(successful)}/{len(self.mics)}")
        
        if successful:
            total_time = max(r.get('execution_time', 0) for r in successful)
            total_elements = sum(r.get('problem_size', 0) for r in successful)
            total_perf = sum(r.get('perf', 0) for r in successful)
            
            print(f"\nMétricas:")
            print(f"  Tiempo total: {total_time:.2f} segundos")
            print(f"  Elementos totales: {total_elements:,}")
            print(f"  Rendimiento total: {total_perf:,.0f} elem/segundo")
            print(f"  Núcleos utilizados: {len(successful) * 60}")
            
            print(f"\nDetalle por MIC:")
            for mic in self.mics:
                mic_results = [r for r in results if r['mic_id'] == mic]
                if mic_results:
                    r = mic_results[0]
                    if r['success']:
                        print(f"  MIC {mic}: {r.get('time', 0):.2f}s, "
                              f"{r.get('perf', 0):,.0f} elem/s, "
                              f"{r.get('problem_size', 0):,} elementos")
                    else:
                        print(f"  MIC {mic}: FALLÓ - {r.get('error', 'Error desconocido')}")
        
        # Guardar resultados
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_mics': len(self.mics),
            'results': results,
            'summary': {
                'successful': len(successful),
                'total_time': total_time if successful else 0,
                'total_elements': total_elements if successful else 0,
                'total_performance': total_perf if successful else 0
            }
        }
        
        filename = f"mic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nResultados guardados en: {filename}")
        return report

def main():
    print("=== SISTEMA PARA 4 INTEL XEON PHI ===")
    print("Núcleos totales: 240 (60 por MIC)")
    print("Memoria total: 32 GB")
    print()
    
    # Crear clúster
    cluster = MICCluster()
    
    # PRUEBA 1: Tamaño pequeño para verificar
    print("PRUEBA 1: Verificación básica")
    print("-" * 40)
    
    test_results = []
    for mic in [0, 1]:  # Solo probar 2 MIC primero
        print(f"\nProbando MIC {mic}...")
        result = cluster.run_on_mic(mic, 1000000)
        test_results.append(result)
        time.sleep(1)  # Pequeña pausa
    
    # PRUEBA 2: Ejecución paralela completa
    print("\n" + "="*60)
    print("PRUEBA 2: Ejecución paralela en las 4 MIC")
    print("="*60)
    
    results = cluster.run_parallel(
        total_size=20000000,  # 20 millones total
        strategy="equal"
    )
    
    # Resumen
    cluster.print_summary(results)
    
    # PRUEBA 3: Carga más grande
    print("\n" + "="*60)
    print("PRUEBA 3: Carga de trabajo grande")
    print("="*60)
    
    results2 = cluster.run_parallel(
        total_size=100000000,  # 100 millones total
        strategy="test"
    )
    
    cluster.print_summary(results2)

if __name__ == "__main__":
    main()

