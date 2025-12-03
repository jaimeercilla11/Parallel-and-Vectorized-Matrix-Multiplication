import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

def crear_matriz(n):
    return np.random.rand(n, n) * 10

def calcular_gflops(n, tiempo):
    return (2.0 * n**3 / tiempo) / 1e9

def mult_basica(A, B):
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def mult_numpy(A, B):
    return np.dot(A, B)

def mult_vectorizada(A, B):
    m, p = A.shape[0], B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        C[i, :] = np.sum(A[i, :, np.newaxis] * B, axis=0)
    return C

def mult_fila(args):
    i, A_fila, B = args
    return i, np.dot(A_fila, B)

def mult_threads(A, B, num_threads):
    m = A.shape[0]
    C = np.zeros((m, B.shape[1]))
    args = [(i, A[i, :], B) for i in range(m)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, fila in executor.map(mult_fila, args):
            C[i, :] = fila
    return C

def mult_bloque(args):
    inicio, fin, A_bloque, B = args
    return inicio, np.dot(A_bloque, B)

def mult_procesos(A, B, num_proc):
    m = A.shape[0]
    C = np.zeros((m, B.shape[1]))
    filas_por_proc = m // num_proc
    args = []
    for i in range(num_proc):
        inicio = i * filas_por_proc
        fin = m if i == num_proc - 1 else (i + 1) * filas_por_proc
        args.append((inicio, fin, A[inicio:fin, :], B))
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        for inicio, bloque in executor.map(mult_bloque, args):
            C[inicio:inicio + bloque.shape[0], :] = bloque
    return C

def ejecutar_pruebas(n):
    print(f"\n{'='*80}\nMATRICES {n}x{n}\n{'='*80}")
    A, B = crear_matriz(n), crear_matriz(n)
    num_cores = cpu_count()
    resultados = []
    
    def medir(nombre, func, *args):
        print(f"Ejecutando: {nombre}...")
        inicio = time.perf_counter()
        func(*args)
        t = time.perf_counter() - inicio
        return t
    
    if n <= 512:
        t_ref = medir("Básico", mult_basica, A, B)
        resultados.append(("Básico (Secuencial)", t_ref, 1.0, 1.0, 1, calcular_gflops(n, t_ref)))
    else:
        t_ref = medir("NumPy (baseline)", mult_numpy, A, B)
    
    t = medir("NumPy", mult_numpy, A, B)
    resultados.append(("NumPy (BLAS/LAPACK)", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    
    if n <= 1024:
        t = medir("Vectorizado", mult_vectorizada, A, B)
        resultados.append(("Vectorizado (NumPy)", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    
    for nt in [2, 4, num_cores]:
        if nt > num_cores:
            continue
        t = medir(f"Threads ({nt})", mult_threads, A, B, nt)
        resultados.append((f"Threads ({nt})", t, t_ref/t, (t_ref/t)/nt, nt, calcular_gflops(n, t)))
    
    for np in [2, num_cores]:
        if np > num_cores:
            continue
        t = medir(f"Procesos ({np})", mult_procesos, A, B, np)
        resultados.append((f"Procesos ({np})", t, t_ref/t, (t_ref/t)/np, np, calcular_gflops(n, t)))
    
    print(f"\n{'='*80}\nRESULTADOS\n{'='*80}")
    print(f"{'Algoritmo':<30} {'Tiempo(s)':>10} {'Speedup':>10} {'Efic(%)':>10} {'Workers':>8} {'GFLOPS':>10}")
    print("-"*80)
    for nombre, tiempo, speedup, eficiencia, workers, gflops in resultados:
        print(f"{nombre:<30} {tiempo:>10.4f} {speedup:>10.2f}x {eficiencia*100:>9.1f}% {workers:>8} {gflops:>10.2f}")
    
    mejor_speedup = max(resultados, key=lambda x: x[2])
    mejor_gflops = max(resultados, key=lambda x: x[5])
    print(f"\n{'='*80}")
    print(f"Mejor Speedup: {mejor_speedup[2]:.2f}x ({mejor_speedup[0]})")
    print(f"Mejor Rendimiento: {mejor_gflops[5]:.2f} GFLOPS ({mejor_gflops[0]})")
    print(f"Memoria usada: {(3 * n * n * 8) / (1024**2):.1f} MB")

def main():
    print("="*80)
    print(" MULTIPLICACIÓN DE MATRICES - ANÁLISIS PARALELO Y VECTORIZADO")
    print("="*80)
    print(f"\nCPUs disponibles: {cpu_count()}")
    
    for n in [256, 512, 1024]:
        ejecutar_pruebas(n)
    
    print(f"\n{'='*80}\nCONCLUSIONES\n{'='*80}")
    print("• NumPy: Mejor rendimiento (usa BLAS/LAPACK optimizado)")
    print("• Threads: Bajo overhead, buen speedup")
    print("• Procesos: Evita GIL pero mayor overhead de comunicación")
    print("• Vectorización: Mejora significativa sobre implementación básica\n")

if __name__ == "__main__":
    main()