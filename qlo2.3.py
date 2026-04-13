import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import gumbel_r, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import base64
from io import StringIO

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="PIV-60 Auto-Calibración",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CLASE DE AUTO-CALIBRACIÓN
# =============================================================================

class CalibradorDinamicoPIV60:
    def __init__(self, df_historial, df_datos):
        self.historial = self._procesar_historial(df_historial)
        self.datos_actuales = self._procesar_datos(df_datos)
        self.config = {}
        
    def _procesar_historial(self, df):
        df.columns = [c.replace('.1', '_2') if '.1' in c else c for c in df.columns]
        historial = []
        cols_nums = [c for c in df.columns if c.startswith('C')]
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

    def _procesar_datos(self, df):
        df.columns = df.columns.str.strip().str.lower()
        datos = {}
        for _, row in df.iterrows():
            datos[int(row['numero'])] = {
                'atraso': int(row['atraso']),
                'frecuencia': int(row['frecuencia'])
            }
        return datos

    def calcular_parametros_gumbel(self):
        atrasos = [info['atraso'] for info in self.datos_actuales.values()]
        loc, scale = gumbel_r.fit(atrasos, floc=0)
        return loc, scale

    def calcular_parametros_gauss(self):
        sumas = [sum(sorteo) for sorteo in self.historial]
        return np.mean(sumas), np.std(sumas)

    def optimizar_pesos_ipc(self):
        if len(self.historial) < 30:
            return 0.25, 0.40, 0.35
            
        def funcion_objetivo(pesos):
            w1, w2, w3 = pesos
            if w1 < 0 or w2 < 0 or w3 < 0 or abs(w1+w2+w3 - 1.0) > 0.01:
                return 1000
            
            score_total = 0
            ventana = min(20, len(self.historial) - 10)
            for i in range(len(self.historial) - ventana, len(self.historial)):
                estado_hasta_i = self._reconstruir_estado(i)
                resultado_real = set(self.historial[i])
                
                scores_num = []
                for n in range(46):
                    if n not in estado_hasta_i: continue
                    at = estado_hasta_i[n]['atraso']
                    fr = estado_hasta_i[n]['frecuencia']
                    
                    max_freq = max(info['frecuencia'] for info in estado_hasta_i.values())
                    f_hist = fr / max_freq if max_freq > 0 else 0
                    v_60 = np.exp(-at / 5.0)
                    t_g = gumbel_r.pdf(at, loc=self.config.get('gumbel_mu', 8.5), 
                                      scale=self.config.get('gumbel_beta', 4.2)) * at
                    
                    ipc_n = w1*f_hist + w2*v_60 + w3*t_g
                    scores_num.append((n, ipc_n))
                
                top_nums = set([x[0] for x in sorted(scores_num, key=lambda x: x[1], reverse=True)[:12]])
                overlap = len(top_nums & resultado_real)
                score_total += overlap
                
            return -score_total

        try:
            res = minimize(funcion_objetivo, x0=[0.25, 0.40, 0.35], 
                           bounds=[(0.1, 0.5), (0.2, 0.6), (0.1, 0.5)],
                           constraints={'type': 'eq', 'fun': lambda x: x[0]+x[1]+x[2]-1.0},
                           method='SLSQP')
            w1, w2, w3 = res.x
            return round(w1, 3), round(w2, 3), round(w3, 3)
        except:
            return 0.25, 0.40, 0.35

    def _reconstruir_estado(self, hasta_indice):
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
        mu, beta = self.calcular_parametros_gumbel()
        g_mean, g_std = self.calcular_parametros_gauss()
        w1, w2, w3 = self.optimizar_pesos_ipc()
        
        Cs_hist = []
        for i in range(60, len(self.historial)):
            estado = self._reconstruir_estado(i)
            C = sum(info['atraso'] for info in estado.values()) + 40
            Cs_hist.append(C)
        
        if Cs_hist:
            p25, p50, p75 = np.percentile(Cs_hist, [25, 50, 75])
        else:
            p25, p50, p75 = 240, 280, 290
        
        self.config = {
            'gumbel_mu': mu, 'gumbel_beta': beta,
            'gauss_mean': g_mean, 'gauss_std': g_std,
            'omega_hist': w1, 'omega_rec': w2, 'omega_gum': w3,
            'zona_inercia': p75, 'zona_equilibrio': (p25, p75), 'zona_ruptura': p25,
            'constante_k': 40
        }
        return self.config

# =============================================================================
# FUNCIONES CORE
# =============================================================================

def distribucion_gumbel_calibrada(atraso, mu, beta):
    if beta <= 0: return 0
    return gumbel_r.pdf(atraso, loc=mu, scale=beta)

def calcular_IPC_calibrado(combinacion, datos, C_actual, config):
    nums = list(combinacion)
    
    freqs = [datos[str(n)]['frecuencia'] for n in nums if str(n) in datos]
    max_freq = max(info['frecuencia'] for info in datos.values())
    F_hist = (np.mean(freqs) / max_freq) if freqs and max_freq > 0 else 0
    
    scores_rec = [np.exp(-datos[str(n)]['atraso'] / 5.0) for n in nums if str(n) in datos]
    V_60 = np.mean(scores_rec) if scores_rec else 0
    
    tensions = []
    for n in nums:
        if str(n) in datos:
            at = datos[str(n)]['atraso']
            tensions.append(distribucion_gumbel_calibrada(at, config['gumbel_mu'], config['gumbel_beta']) * at)
    T_g = np.sum(tensions)
    
    suma = sum(nums)
    diff = abs(suma - config['gauss_mean'])
    phi_gauss = 0.5 * (diff / config['gauss_std']) ** 2 if config['gauss_std'] > 0 else 0
    
    ipc = config['omega_hist']*F_hist + config['omega_rec']*V_60 + config['omega_gum']*T_g - phi_gauss
    
    return {'IPC_Total': ipc, 'F_hist': F_hist, 'V_60': V_60, 'T_g': T_g, 'phi_gauss': phi_gauss, 'suma': suma}

# =============================================================================
# INTERFAZ STREAMLIT
# =============================================================================

st.title("🔬 Protocolo PIV-60 v4.0 - Auto-Calibración Dinámica")
st.markdown("**Documento:** PIP-2026-X46 | **Clasificación:** Informe Técnico de Alta Precisión")

# Sidebar
with st.sidebar:
    st.header("📁 Carga de Archivos")
    st.info("Sube ambos archivos CSV para comenzar")
    
    archivo_historial = st.file_uploader("Historial_Tradicional.csv", type=['csv'], key='hist')
    archivo_datos = st.file_uploader("datos_actuales.csv", type=['csv'], key='datos')
    
    st.markdown("---")
    st.markdown("**Parámetros:**")
    top_n = st.slider("Top combinaciones a mostrar", 5, 100, 15)
    mostrar_todas = st.checkbox("Mostrar todas las combinaciones filtradas", value=True)
    
    ejecutar = st.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True)

if ejecutar:
    if archivo_historial is None or archivo_datos is None:
        st.error("❌ Por favor, sube ambos archivos CSV")
        st.stop()
    
    try:
        with st.spinner("📥 Cargando y procesando datos..."):
            df_hist = pd.read_csv(archivo_historial)
            df_datos = pd.read_csv(archivo_datos)
            
        with st.spinner("⚙️ Ejecutando auto-calibración dinámica..."):
            calibrador = CalibradorDinamicoPIV60(df_hist, df_datos)
            config = calibrador.ejecutar_calibracion()
            
        # Mostrar configuración calibrada
        st.success("✅ Auto-calibración completada")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Gumbel μ", f"{config['gumbel_mu']:.2f}")
            st.metric("Gumbel β", f"{config['gumbel_beta']:.2f}")
        with col2:
            st.metric("Gauss μ", f"{config['gauss_mean']:.1f}")
            st.metric("Gauss σ", f"{config['gauss_std']:.1f}")
        with col3:
            st.metric("ω Histórico", f"{config['omega_hist']:.3f}")
            st.metric("ω Reciente", f"{config['omega_rec']:.3f}")
            st.metric("ω Gumbel", f"{config['omega_gum']:.3f}")
        
        # Generar combinaciones
        with st.spinner("🔄 Generando combinaciones PIV-60..."):
            datos = calibrador.datos_actuales
            historial = calibrador.historial
            
            momento = [n for n, info in datos.items() if info['atraso'] == 0]
            masa = [n for n, info in datos.items() if 1 <= info['atraso'] <= 9]
            tension = [n for n, info in datos.items() if info['atraso'] > 15]
            
            suma_atrasos = sum(info['atraso'] for info in datos.values())
            C_actual = suma_atrasos + config['constante_k']
            
            st.info(f"📊 Clasificación PIV-60: Momento={len(momento)}, Masa Crítica={len(masa)}, Tensión Crítica={len(tension)}")
            
            combinaciones = []
            for m in momento:
                for mc in combinations(masa, 4):
                    for t in tension:
                        combo = tuple(sorted([m] + list(mc) + [t]))
                        if len(set(combo)) == 6:
                            combinaciones.append(combo)
            
            st.write(f"✅ {len(combinaciones):,} combinaciones candidatas generadas bajo regla 1-4-1")
            
            # Filtrar y rankear TODAS las combinaciones
            resultados = []
            for combo in combinaciones:
                suma_combo = sum(combo)
                if not (config['gauss_mean'] - 2.5*config['gauss_std'] <= suma_combo <= config['gauss_mean'] + 2.5*config['gauss_std']):
                    continue
                    
                atrasos_c = [datos[n]['atraso'] for n in combo]
                S = C_actual - sum(atrasos_c)
                
                if S > config['zona_inercia']:
                    zona, peso_z = 'INERCIA', 1.0
                elif config['zona_equilibrio'][0] < S < config['zona_equilibrio'][1]:
                    zona, peso_z = 'EQUILIBRIO', 1.5
                elif S < config['zona_ruptura']:
                    zona, peso_z = 'RUPTURA', 1.2
                else:
                    zona, peso_z = 'TRANSICIÓN', 0.8
                
                ipc_d = calcular_IPC_calibrado(combo, datos, C_actual, config)
                score = ipc_d['IPC_Total'] * peso_z
                
                resultados.append({
                    'Rank': 0,
                    'Combinación': ' - '.join(map(str, combo)),
                    'N1': combo[0], 'N2': combo[1], 'N3': combo[2], 
                    'N4': combo[3], 'N5': combo[4], 'N6': combo[5],
                    'Score': score, 'IPC': ipc_d['IPC_Total'],
                    'S': S, 'Zona': zona,
                    'Suma': d['suma'],
                    'F_Hist': d['F_hist'],
                    'V_60': d['V_60'],
                    'T_Gumbel': d['T_g'],
                    'Penal_Gauss': d['phi_gauss']
                })
            
            # Ordenar
            resultados.sort(key=lambda x: x['Score'], reverse=True)
            
            # Asignar ranks
            for i, r in enumerate(resultados, 1):
                r['Rank'] = i
            
            finalistas = resultados[:top_n]
            
        # ========== MOSTRAR TOP COMBINACIONES ==========
        st.subheader(f"🏆 Top {top_n} Combinaciones Recomendadas")
        
        for i, item in enumerate(finalistas, 1):
            with st.expander(f"#{i:02d} | {item['Combinación']} | Score: {item['Score']:.4f}", expanded=(i<=5)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Estado Energético:** {item['Zona']}")
                    st.write(f"**Transición S:** {item['S']:.1f}")
                    st.write(f"**Suma:** {item['Suma']}")
                with col_b:
                    st.write(f"**IPC Total:** {item['IPC']:.4f}")
                    st.write(f"**IPC Histórico:** {item['F_Hist']:.3f}")
                    st.write(f"**IPC Reciente:** {item['V_60']:.3f}")
                    st.write(f"**IPC Gumbel:** {item['T_Gumbel']:.3f}")
        
        # ========== TABLA COMPLETA DE COMBINACIONES ==========
        if mostrar_todas:
            st.subheader(f"📋 Todas las Combinaciones Filtradas ({len(resultados):,})")
            st.info("💡 Las combinaciones están ordenadas por Score (mejor a peor)")
            
            # Crear DataFrame
            df_resultados = pd.DataFrame(resultados)
            
            # Mostrar tabla interactiva
            st.dataframe(
                df_resultados[['Rank', 'Combinación', 'Score', 'IPC', 'Zona', 'Suma', 'S']],
                use_container_width=True,
                height=600
            )
            
            # Botón de descarga CSV
            csv = df_resultados.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 Descargar {len(resultados):,} Combinaciones (CSV)",
                data=csv,
                file_name=f"PIV60_Combinaciones_{len(resultados)}_resultados.csv",
                mime="text/csv",
            )
            
            # Estadísticas
            st.subheader("📊 Estadísticas del Lote")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Combinaciones", f"{len(resultados):,}")
            with col_s2:
                st.metric("Score Máximo", f"{df_resultados['Score'].max():.4f}")
            with col_s3:
                st.metric("Score Promedio", f"{df_resultados['Score'].mean():.4f}")
            with col_s4:
                st.metric("Score Mínimo", f"{df_resultados['Score'].min():.4f}")
            
            # Distribución por zona
            st.subheader(" Distribución por Zona Energética")
            col_z1, col_z2, col_z3, col_z4 = st.columns(4)
            zonas = df_resultados['Zona'].value_counts()
            with col_z1:
                st.metric("EQUILIBRIO", f"{zonas.get('EQUILIBRIO', 0):,}")
            with col_z2:
                st.metric("INERCIA", f"{zonas.get('INERCIA', 0):,}")
            with col_z3:
                st.metric("RUPTURA", f"{zonas.get('RUPTURA', 0):,}")
            with col_z4:
                st.metric("TRANSICIÓN", f"{zonas.get('TRANSICIÓN', 0):,}")
        
        # Mapa de calor
        st.subheader("🎨 Mapa de Calor de Atrasos")
        numeros = sorted(list(datos.keys()))
        matriz = np.full((6, 8), np.nan)
        for i, n in enumerate(numeros):
            matriz[i//8, i%8] = datos[n]['atraso']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(matriz, annot=True, fmt='.0f', cmap='RdYlGn_r', linewidths=1, 
                   linecolor='gray', ax=ax, cbar_kws={'label': 'Días de Atraso'})
        ax.set_xticks([])
        ax.set_yticks([])
        
        for i, n in enumerate(numeros):
            r, c = i//8, i%8
            ax.text(c+0.5, r+0.8, str(n), ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=10)
        
        st.pyplot(fig)
        
        # Estadísticas del sistema
        st.subheader("📊 Estadísticas del Sistema")
        atrasos_vals = [info['atraso'] for info in datos.values()]
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Números en Momento (A=0)", len(momento))
        with col_s2:
            st.metric("Masa Crítica (1-9)", len(masa))
        with col_s3:
            st.metric("Tensión Crítica (>15)", len(tension))
        with col_s4:
            st.metric("Atraso Máximo", max(atrasos_vals))
        
        st.success(f"✅ Análisis completado exitosamente")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
else:
    st.info("👈 Sube los archivos y presiona 'Ejecutar Análisis' para comenzar")

# Footer
st.markdown("---")
st.caption("Protocolo PIV-60 v4.0 | Ingeniería Probabilística | Auto-Calibración Dinámica")
