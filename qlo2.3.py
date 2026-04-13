#!/usr/bin/env python3
# =============================================================================
# PROTOCOLO PIV-60 v4.0: AUTO-CALIBRACIÓN DINÁMICA
# Documento: PIP-2026-X46 | Clasificación: Informe Técnico de Alta Precisión
# =============================================================================
# Deps: pip install pandas numpy scipy matplotlib seaborn
# =============================================================================

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import gumbel_r, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time

warnings.filterwarnings("ignore")

# =============================================================================
# 1. MÓDULO DE AUTO-CALIBRACIÓN DINÁMICA
# =============================================================================

class CalibradorDinamicoPIV60:
    """
    Ajusta automáticamente los parámetros del protocolo PIV-60
    utilizando métodos estadísticos y optimización walk-forward.
    """
    
    def __init__(self, archivo_historial, archivo_datos_actuales):
        self.historial = self._cargar_historial(archivo_historial)
        self.datos_actuales = self._cargar_datos_actuales(archivo_datos_actuales)
        self.config = {}
        
    def _cargar_historial(self, path):
        """Carga historial manejando headers duplicados y desorden."""
        df = pd.read_csv(path, header=0)
        # Corregir header duplicado C3 -> C3.1
        df.columns = [c.replace('.1', '_2') if '.1' in c else c for c in df.columns]
        cols_nums = [c for c in df.columns if c.startswith('C')]
        historial = []
        for _, row in df.iterrows():
            nums = set()
            for c in cols_nums:
                try:
                    n = int(row[c])
                    if 0 <= n <= 45:
                        nums.add(n)
                except:
                    continue
            if len(nums) >= 5:
                historial.append(sorted(nums))
        return historial

    def _cargar_datos_actuales(self, path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        datos = {}
        for _, row in df.iterrows():
            datos[int(row['numero'])] = {
                'atraso': int(row['atraso']),
                'frecuencia': int(row['frecuencia'])
            }
        return datos

    def calcular_parametros_gumbel(self):
        """Ajuste MLE de la distribución de Gumbel sobre atrasos actuales."""
        atrasos = [info['atraso'] for info in self.datos_actuales.values()]
        loc, scale = gumbel_r.fit(atrasos, floc=0) # floc=0 fuerza inicio en 0
        return loc, scale

    def calcular_parametros_gauss(self):
        """Media y desviación de las sumas históricas."""
        sumas = [sum(sorteo) for sorteo in self.historial]
        return np.mean(sumas), np.std(sumas)

    def optimizar_pesos_ipc(self):
        """
        Walk-Forward Optimization sobre los últimos 20 sorteos.
        Maximiza el solapamiento esperado entre top-numeros predichos y reales.
        """
        print("⚙️ Optimizando pesos IPC (Walk-Forward 20 sorteos)...")
        
        def funcion_objetivo(pesos):
            w1, w2, w3 = pesos
            if w1 < 0 or w2 < 0 or w3 < 0 or abs(w1+w2+w3 - 1.0) > 0.01:
                return 1000 # Penalización fuerte
            
            score_total = 0
            ventana = 20
            for i in range(len(self.historial) - ventana, len(self.historial)):
                # Reconstruir estado hasta i
                estado_hasta_i = self._reconstruir_estado(i)
                resultado_real = set(self.historial[i])
                
                # Calcular score por número (proxy rápido)
                scores_num = []
                for n in range(46):
                    if n not in estado_hasta_i: continue
                    at = estado_hasta_i[n]['atraso']
                    fr = estado_hasta_i[n]['frecuencia']
                    
                    # Componentes normalizados
                    f_hist = fr / max(info['frecuencia'] for info in estado_hasta_i.values())
                    v_60 = np.exp(-at / 5.0)
                    t_g = gumbel_r.pdf(at, loc=self.config['gumbel_mu'], scale=self.config['gumbel_beta']) * at
                    
                    ipc_n = w1*f_hist + w2*v_60 + w3*t_g
                    scores_num.append((n, ipc_n))
                
                # Top 12 números por IPC
                top_nums = set([x[0] for x in sorted(scores_num, key=lambda x: x[1], reverse=True)[:12]])
                overlap = len(top_nums & resultado_real)
                score_total += overlap
                
            return -score_total # Minimizar negativo = maximizar aciertos

        # Optimización restringida
        res = minimize(funcion_objetivo, x0=[0.25, 0.40, 0.35], 
                       bounds=[(0.1, 0.5), (0.2, 0.6), (0.1, 0.5)],
                       constraints={'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]-1.0})
        
        w1, w2, w3 = res.x
        return round(w1, 3), round(w2, 3), round(w3, 3)

    def _reconstruir_estado(self, hasta_indice):
        """Reconstruye atrasos y frecuencias hasta un punto histórico."""
        atrasos = {i: 0 for i in range(46)}
        frecuencias = {i: 0 for i in range(46)}
        
        for s in self.historial[:hasta_indice]:
            for n in s:
                frecuencias[n] += 1
                atrasos[n] = 0
            for n in range(46):
                if n not in s:
                    atrasos[n] += 1
                    
        return {n: {'atraso': atrasos[n], 'frecuencia': frecuencias[n]} for n in range(46)}

    def ejecutar_calibracion(self):
        print("\n🔬 INICIANDO AUTO-CALIBRACIÓN DINÁMICA v4.0")
        print("="*50)
        
        # 1. Gumbel
        mu, beta = self.calcular_parametros_gumbel()
        print(f"📈 Gumbel ajustado (MLE): μ={mu:.2f}, β={beta:.2f}")
        
        # 2. Gaussiano
        g_mean, g_std = self.calcular_parametros_gauss()
        print(f"📊 Gaussiano calibrado: μ_suma={g_mean:.1f}, σ_suma={g_std:.1f}")
        
        # 3. Pesos IPC
        w1, w2, w3 = self.optimizar_pesos_ipc()
        print(f"⚖️ Pesos IPC optimizados: ω_hist={w1}, ω_rec={w2}, ω_gum={w3}")
        
        # 4. Zonas de Transición (ajuste percentílico)
        Cs_hist = []
        for i in range(60, len(self.historial)):
            estado = self._reconstruir_estado(i)
            C = sum(info['atraso'] for info in estado.values()) + 40
            Cs_hist.append(C)
        p25, p50, p75 = np.percentile(Cs_hist, [25, 50, 75])
        
        self.config = {
            'gumbel_mu': mu, 'gumbel_beta': beta,
            'gauss_mean': g_mean, 'gauss_std': g_std,
            'omega_hist': w1, 'omega_rec': w2, 'omega_gum': w3,
            'zona_inercia': p75, 'zona_equilibrio': (p25, p75), 'zona_ruptura': p25,
            'constante_k': 40
        }
        print("✅ Auto-calibración completada.\n")
        return self.config

# =============================================================================
# 2. LÓGICA CORE PIV-60 (USANDO CONFIG CALIBRADA)
# =============================================================================

def distribucion_gumbel_calibrada(atraso, mu, beta):
    if beta <= 0: return 0
    return gumbel_r.pdf(atraso, loc=mu, scale=beta)

def calcular_transicion_energetica(C, atrasos_combo):
    return C - sum(atrasos_combo)

def evaluar_zona_calibrada(S, config):
    if S > config['zona_inercia']:
        return 'INERCIA', 1.0
    elif config['zona_equilibrio'][0] < S < config['zona_equilibrio'][1]:
        return 'EQUILIBRIO', 1.5
    elif S < config['zona_ruptura']:
        return 'RUPTURA', 1.2
    return 'TRANSICIÓN', 0.8

def calcular_IPC_calibrado(combinacion, datos, C_actual, config):
    nums = list(combinacion)
    
    # F_hist
    freqs = [datos[str(n)]['frecuencia'] for n in nums if str(n) in datos]
    max_freq = max(info['frecuencia'] for info in datos.values())
    F_hist = (np.mean(freqs) / max_freq) if freqs else 0
    
    # V_60
    scores_rec = [np.exp(-datos[str(n)]['atraso'] / 5.0) for n in nums if str(n) in datos]
    V_60 = np.mean(scores_rec) if scores_rec else 0
    
    # T_g (calibrado)
    tensions = []
    for n in nums:
        if str(n) in datos:
            at = datos[str(n)]['atraso']
            tensions.append(distribucion_gumbel_calibrada(at, config['gumbel_mu'], config['gumbel_beta']) * at)
    T_g = np.sum(tensions)
    
    # Gauss (calibrado)
    suma = sum(nums)
    diff = abs(suma - config['gauss_mean'])
    phi_gauss = 0.5 * (diff / config['gauss_std']) ** 2
    
    ipc = config['omega_hist']*F_hist + config['omega_rec']*V_60 + config['omega_gum']*T_g - phi_gauss
    
    return {'IPC_Total': ipc, 'F_hist': F_hist, 'V_60': V_60, 'T_g': T_g, 'phi_gauss': phi_gauss, 'suma': suma}

def ejecutar_piv60_calibrado(archivo_datos, config, top_n=15):
    print("🔄 Generando combinaciones con parámetros calibrados...")
    df = pd.read_csv(archivo_datos)
    df.columns = df.columns.str.strip().str.lower()
    datos = {str(int(row['numero'])): {'atraso': int(row['atraso']), 'frecuencia': int(row['frecuencia'])} for _, row in df.iterrows()}
    
    momento = [int(n) for n, info in datos.items() if info['atraso'] == 0]
    masa = [int(n) for n, info in datos.items() if 1 <= info['atraso'] <= 9]
    tension = [int(n) for n, info in datos.items() if info['atraso'] > 15]
    
    suma_atrasos = sum(info['atraso'] for info in datos.values())
    C_actual = suma_atrasos + config['constante_k']
    
    combinaciones = []
    for m in momento:
        for mc in combinations(masa, 4):
            for t in tension:
                combo = tuple(sorted([m] + list(mc) + [t]))
                if len(set(combo)) == 6:
                    combinaciones.append(combo)
                    
    print(f"✅ {len(combinaciones):,} candidatas generadas. Aplicando filtros calibrados...")
    resultados = []
    for combo in combinaciones:
        if not (config['gauss_mean'] - 2.5*config['gauss_std'] <= sum(combo) <= config['gauss_mean'] + 2.5*config['gauss_std']):
            continue
            
        atrasos_c = [datos[str(n)]['atraso'] for n in combo]
        S = calcular_transicion_energetica(C_actual, atrasos_c)
        zona, peso_z = evaluar_zona_calibrada(S, config)
        ipc_d = calcular_IPC_calibrado(combo, datos, C_actual, config)
        
        score = ipc_d['IPC_Total'] * peso_z
        resultados.append({'combo': combo, 'score': score, 'S': S, 'zona': zona, 'ipc': ipc_d})
        
    resultados.sort(key=lambda x: x['score'], reverse=True)
    return resultados[:top_n], datos

# =============================================================================
# 3. VISUALIZACIÓN Y SALIDA
# =============================================================================

def generar_mapa_calor(datos, config):
    numeros = sorted([int(k) for k in datos.keys()])
    matriz = np.full((6, 8), np.nan)
    for i, n in enumerate(numeros):
        matriz[i//8, i%8] = datos[str(n)]['atraso']
        
    plt.figure(figsize=(14, 7))
    plt.suptitle("TABLERO DE ATRASOS - ESTADO CALIBRADO PIV-60 v4.0", fontsize=15, fontweight='bold')
    ax = sns.heatmap(matriz, annot=True, fmt='.0f', cmap='RdYlGn_r', linewidths=1, linecolor='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for i, n in enumerate(numeros):
        r, c = i//8, i%8
        ax.text(c+0.5, r+0.8, str(n), ha='center', va='center', color='black', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig("mapa_calor_calibrado.png", dpi=150)
    print("🎨 Mapa de calor guardado: 'mapa_calor_calibrado.png'")

def imprimir_resultados_calibrados(finalistas, config):
    print("\n" + "="*60)
    print("🏆 TOP 15 COMBINACIONES (PARAMETROS AUTO-CALIBRADOS)")
    print("="*60)
    for i, item in enumerate(finalistas, 1):
        d = item['ipc']
        print(f"\n🥇 #{i:02d} | {' - '.join(map(str, item['combo']))}")
        print(f"    ├─ 📊 Score: {item['score']:.4f} | ⚡ S={item['S']:.1f} [{item['zona']}]")
        print(f"    ├─ 🧩 IPC: Hist({config['omega_hist']:.2f})={d['F_hist']:.3f} | "
              f"Rec({config['omega_rec']:.2f})={d['V_60']:.3f} | Gum({config['omega_gum']:.2f})={d['T_g']:.3f}")
        print(f"    └─  Suma: {d['suma']} | φ_Gauss: -{d['phi_gauss']:.3f}")

# =============================================================================
# 4. EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    start = time.time()
    try:
        # 1. Auto-calibración
        calibrador = CalibradorDinamicoPIV60("Historial_Tradicional.csv", "datos_actuales.csv")
        config_calibrada = calibrador.ejecutar_calibracion()
        
        # 2. Predicción
        finalistas, datos = ejecutar_piv60_calibrado("datos_actuales.csv", config_calibrada)
        
        # 3. Salida
        imprimir_resultados_calibrados(finalistas, config_calibrada)
        generar_mapa_calor(datos, config_calibrada)
        
        print(f"\n⏱️ Tiempo total: {time.time()-start:.2f}s")
        print("✅ Proceso finalizado con éxito.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
