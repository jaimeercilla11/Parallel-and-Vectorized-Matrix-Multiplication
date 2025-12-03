import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import psutil

def crear_matriz(n):
    return np.random.rand(n, n) * 10

def medir_tiempo(func, *args):
    inicio = time.perf_counter()
    resultado = func(*args)
    return resultado, time.perf_counter() - inicio

def calcular_gflops(n, tiempo):
    return (2.0 * n**3 / tiempo) / 1e9

# 1. Básico
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
    i, A_fila, B, p, n = args
    resultado = np.zeros(p)
    for j in range(p):
        for k in range(n):
            resultado[j] += A_fila[k] * B[k, j]
    return i, resultado

def mult_threads(A, B, num_threads=None):
    if num_threads is None:
        num_threads = cpu_count()
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((m, p))
    args = [(i, A[i, :], B, p, n) for i in range(m)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, fila in executor.map(mult_fila, args):
            C[i, :] = fila
    return C

def mult_bloque_proc(args):
    inicio, fin, A_bloque, B = args
    m, n, p = fin - inicio, A_bloque.shape[1], B.shape[1]
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A_bloque[i, k] * B[k, j]
    return inicio, C

def mult_procesos(A, B, num_proc=None):
    if num_proc is None:
        num_proc = cpu_count()
    m = A.shape[0]
    C = np.zeros((m, B.shape[1]))
    filas_por_proc = m // num_proc
    args = []
    for i in range(num_proc):
        inicio = i * filas_por_proc
        fin = m if i == num_proc - 1 else (i + 1) * filas_por_proc
        args.append((inicio, fin, A[inicio:fin, :], B))
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        for inicio, bloque in executor.map(mult_bloque_proc, args):
            C[inicio:inicio + bloque.shape[0], :] = bloque
    return C

def mult_tiling(A, B, tam_bloque=64):
    m, n, p = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((m, p))
    for ii in range(0, m, tam_bloque):
        for jj in range(0, p, tam_bloque):
            for kk in range(0, n, tam_bloque):
                i_max, j_max, k_max = min(ii + tam_bloque, m), min(jj + tam_bloque, p), min(kk + tam_bloque, n)
                for i in range(ii, i_max):
                    for j in range(jj, j_max):
                        for k in range(kk, k_max):
                            C[i, j] += A[i, k] * B[k, j]
    return C

def mult_bloque_numpy(args):
    inicio, fin, A_bloque, B = args
    return inicio, np.dot(A_bloque, B)

def mult_paralela_numpy(A, B, num_proc=None):
    if num_proc is None:
        num_proc = cpu_count()
    m = A.shape[0]
    C = np.zeros((m, B.shape[1]))
    filas_por_proc = m // num_proc
    args = []
    for i in range(num_proc):
        inicio = i * filas_por_proc
        fin = m if i == num_proc - 1 else (i + 1) * filas_por_proc
        args.append((inicio, fin, A[inicio:fin, :], B))
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        for inicio, bloque in executor.map(mult_bloque_numpy, args):
            C[inicio:inicio + bloque.shape[0], :] = bloque
    return C

def ejecutar_pruebas(n):
    print(f"\n{'='*90}\nMATRICES {n}x{n}\n{'='*90}")
    A, B = crear_matriz(n), crear_matriz(n)
    resultados = []
    num_cores = cpu_count()
    
    if n <= 512:
        print("[1/7] Básico...")
        _, t = medir_tiempo(mult_basica, A, B)
        resultados.append(("Básico (Secuencial)", t, 1.0, 1.0, 1, calcular_gflops(n, t)))
        t_ref = t
    else:
        print("[1/7] Básico [Omitido]")
        t_ref = None
    
    print("[2/7] NumPy...")
    _, t = medir_tiempo(mult_numpy, A, B)
    if not t_ref:
        t_ref = t
    resultados.append(("NumPy (BLAS/LAPACK)", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    
    if n <= 1024:
        print("[3/7] Vectorizado...")
        _, t = medir_tiempo(mult_vectorizada, A, B)
        resultados.append(("Vectorizado (NumPy)", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    else:
        print("[3/7] Vectorizado [Omitido]")
    
    print("[4/7] Threads...")
    for nt in [2, 4, num_cores]:
        if nt > num_cores:
            continue
        _, t = medir_tiempo(mult_threads, A, B, nt)
        resultados.append((f"Paralelo Threads ({nt} threads)", t, t_ref/t, (t_ref/t)/nt, nt, calcular_gflops(n, t)))
    
    print("[5/7] Procesos...")
    for np in [2, num_cores]:
        if np > num_cores:
            continue
        _, t = medir_tiempo(mult_procesos, A, B, np)
        resultados.append((f"Paralelo Procesos ({np} procesos)", t, t_ref/t, (t_ref/t)/np, np, calcular_gflops(n, t)))
    
    if n <= 512:
        print("[6/7] Tiling...")
        _, t = medir_tiempo(mult_tiling, A, B, 64)
        resultados.append(("Optimizado (Tiling)", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    else:
        print("[6/7] Tiling [Omitido]")
    
    print("[7/7] Paralelo + NumPy...")
    _, t = medir_tiempo(mult_paralela_numpy, A, B, num_cores)
    resultados.append((f"Paralelo + NumPy ({num_cores} procesos)", t, t_ref/t, (t_ref/t)/num_cores, num_cores, calcular_gflops(n, t)))
    
    print(f"\n{'='*90}\nRESULTADOS\n{'='*90}")
    print(f"{'Algoritmo':<35} {'Tiempo (s)':>12} {'Speedup':>10} {'Eficiencia':>12} {'Threads':>8} {'GFLOPS':>10}")
    print("-"*90)
    for nombre, tiempo, speedup, eficiencia, threads, gflops in resultados:
        print(f"{nombre:<35} {tiempo:>12.4f} {speedup:>10.2f}x {eficiencia*100:>11.1f}% {threads:>8} {gflops:>10.2f}")
    
    # Análisis
    mejor_speedup = max(resultados, key=lambda x: x[2])
    mejor_gflops = max(resultados, key=lambda x: x[5])
    print(f"\n{'='*90}\nANÁLISIS\n{'='*90}")
    print(f"• Mejor Speedup: {mejor_speedup[2]:.2f}x ({mejor_speedup[0]})")
    print(f"• Mejor Rendimiento: {mejor_gflops[5]:.2f} GFLOPS ({mejor_gflops[0]})")
    print(f"• Memoria: {(3 * n * n * 8) / (1024**2):.1f} MB")

def main():
    print("="*90)
    print("    MULTIPLICACIÓN DE MATRICES PARALELA Y VECTORIZADA")
    print("="*90)
    print(f"\nCPUs: {cpu_count()} | RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    for n in [256, 512, 1024]:
        ejecutar_pruebas(n)
    
    print(f"\n{'='*90}\nCONCLUSIONES\n{'='*90}")
    print("• NumPy: mejor rendimiento (BLAS/LAPACK)")
    print("• Threads: efectivo con bajo overhead")
    print("• Procesos: supera GIL pero overhead comunicación")
    print("• Tiling: mejora cache locality")
    print("• Paralelo+NumPy: excelente balance\n")

if __name__ == "__main__":
    main()