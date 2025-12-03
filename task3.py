import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

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

def ejecutar_pruebas(n, resultados_globales):
    print(f"\n{'='*80}\nMATRICES {n}x{n}\n{'='*80}")
    A, B = crear_matriz(n), crear_matriz(n)
    num_cores = cpu_count()
    resultados = []
    
    def medir(nombre, func, *args):
        print(f"Executing: {nombre}...")
        inicio = time.perf_counter()
        func(*args)
        t = time.perf_counter() - inicio
        return t
    
    if n <= 512:
        t_ref = medir("Basic", mult_basica, A, B)
        resultados.append(("Basic", t_ref, 1.0, 1.0, 1, calcular_gflops(n, t_ref)))
    else:
        t_ref = medir("NumPy (baseline)", mult_numpy, A, B)
    
    t = medir("NumPy", mult_numpy, A, B)
    resultados.append(("NumPy", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    
    if n <= 1024:
        t = medir("Vectorized", mult_vectorizada, A, B)
        resultados.append(("Vectorized", t, t_ref/t, t_ref/t, 1, calcular_gflops(n, t)))
    
    for nt in [2, 4, num_cores]:
        if nt > num_cores:
            continue
        t = medir(f"Threads ({nt})", mult_threads, A, B, nt)
        resultados.append((f"Threads-{nt}", t, t_ref/t, (t_ref/t)/nt, nt, calcular_gflops(n, t)))
    
    for np in [2, num_cores]:
        if np > num_cores:
            continue
        t = medir(f"Processes ({np})", mult_procesos, A, B, np)
        resultados.append((f"Processes-{np}", t, t_ref/t, (t_ref/t)/np, np, calcular_gflops(n, t)))
    
    resultados_globales[n] = resultados
    
    print(f"\n{'='*80}\nRESULTS\n{'='*80}")
    print(f"{'Algorithm':<30} {'Time(s)':>10} {'Speedup':>10} {'Eff(%)':>10} {'Workers':>8} {'GFLOPS':>10}")
    print("-"*80)
    for nombre, tiempo, speedup, eficiencia, workers, gflops in resultados:
        print(f"{nombre:<30} {tiempo:>10.4f} {speedup:>10.2f}x {eficiencia*100:>9.1f}% {workers:>8} {gflops:>10.2f}")
    
    mejor_speedup = max(resultados, key=lambda x: x[2])
    mejor_gflops = max(resultados, key=lambda x: x[5])
    print(f"\n{'='*80}")
    print(f"Best Speedup: {mejor_speedup[2]:.2f}x ({mejor_speedup[0]})")
    print(f"Best Performance: {mejor_gflops[5]:.2f} GFLOPS ({mejor_gflops[0]})")
    print(f"Memory used: {(3 * n * n * 8) / (1024**2):.1f} MB")

def generar_graficas(resultados_globales):
    print(f"\n{'='*80}\nGENERATING GRAPHS...\n{'='*80}")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'Basic': '#1f77b4',
        'NumPy': '#ff7f0e',
        'Vectorized': '#2ca02c',
        'Threads-2': '#d62728',
        'Threads-4': '#9467bd',
        'Threads-8': '#8c564b',
        'Processes-2': '#e377c2',
        'Processes-8': '#7f7f7f'
    }
    
    for alg in ['NumPy', 'Threads-4', 'Threads-8', 'Processes-8']:
        sizes = []
        gflops = []
        for n in sorted(resultados_globales.keys()):
            for nombre, _, _, _, _, gf in resultados_globales[n]:
                if nombre == alg:
                    sizes.append(n)
                    gflops.append(gf)
        if sizes:
            ax1.plot(sizes, gflops, marker='o', label=alg, linewidth=3,
                    color=colors.get(alg, 'gray'), markersize=10)
    ax1.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
    ax1.set_ylabel('GFLOPS', fontsize=13, fontweight='bold')
    ax1.set_title('Performance (GFLOPS) vs Matrix Size', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(sorted(resultados_globales.keys()))
    
    n_medio = 512
    if n_medio in resultados_globales:
        workers_t = []
        speedup_t = []
        workers_p = []
        speedup_p = []
        for nombre, _, speedup, _, workers, _ in resultados_globales[n_medio]:
            if 'Threads' in nombre:
                workers_t.append(workers)
                speedup_t.append(speedup)
            elif 'Processes' in nombre:
                workers_p.append(workers)
                speedup_p.append(speedup)
        
        ax2.plot(workers_t, speedup_t, marker='o', label='Threads', 
                linewidth=3, markersize=10, color='#d62728')
        ax2.plot(workers_p, speedup_p, marker='s', label='Processes', 
                linewidth=3, markersize=10, color='#e377c2')

        max_workers = max(max(workers_t), max(workers_p))
        ax2.plot([1, max_workers], [1, max_workers], '--', color='gray', 
                label='Ideal Speedup', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Number of Workers', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Speedup', fontsize=13, fontweight='bold')
        ax2.set_title(f'Speedup Scaling (512×512 matrices)', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matrix_multiplication_analysis.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'matrix_multiplication_analysis.png'")
    plt.show()

def main():
    print("="*80)
    print(" MATRIX MULTIPLICATION - PARALLEL AND VECTORIZED ANALYSIS")
    print("="*80)
    print(f"\nAvailable CPUs: {cpu_count()}")
    
    resultados_globales = {}
    
    for n in [256, 512, 1024]:
        ejecutar_pruebas(n, resultados_globales)
    
    print(f"\n{'='*80}\nCONCLUSIONS\n{'='*80}")
    print("• NumPy: Best performance (uses optimized BLAS/LAPACK)")
    print("• Threads: Low overhead, good speedup")
    print("• Processes: Avoids GIL but higher communication overhead")
    print("• Vectorization: Significant improvement over basic implementation\n")
    
    # Generar gráficas
    generar_graficas(resultados_globales)

if __name__ == "__main__":
    main()