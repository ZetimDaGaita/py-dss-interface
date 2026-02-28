import py_dss_interface
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"
out_excel_final = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\Resultados_Caso1_Completo.xlsx"
V_MIN, V_MAX = 0.93, 1.05  # PRODIST limites

# pasta para figuras
figures_dir = os.path.join(os.path.dirname(out_excel_final), "figures")
os.makedirs(figures_dir, exist_ok=True)

# -----------------------------
# INICIALIZAÇÃO DO OpenDSS
# -----------------------------
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")

# desativa reguladores (Caso 1)
for i in range(1, 4):
    try:
        dss.text(f"disable regcontrol.reg{i}")
    except Exception:
        pass
    try:
        dss.text(f"edit transformer.reg{i} taps=[1.625 1.625 1.625]")
    except Exception:
        pass
dss.text("set controlmode=off")


dss.text("New Monitor.Mon_Substation element=Transformer.XFM1 terminal=1 mode=1")
dss.monitors.name = "Mon_Substation"

# Garante que TODOS os bancos de capacitores estejam ligados
for cap in dss.capacitors.names:
    try:
        dss.text(f"edit capacitor.{cap} enabled=true")
    except Exception:
        pass


# quick initial solve snapshot
dss.text("solve")

print("\n==============================")
print("ESTADO DOS CAPACITORES")
print("==============================")

for cap in dss.capacitors.names:
    dss.circuit.set_active_element(f"Capacitor.{cap}")
    powers = dss.cktelement.powers  # [P1, Q1, P2, Q2, ...]
    q_inj = -sum(powers[1::2])      # Q injetado (kvar), sinal físico
    print(f"{cap}: Q_injetado = {q_inj:.2f} kvar")

print("\n==============================")
print("TENSÕES POR BARRA (pu)")
print("==============================")

for bus in ["611", "684", "671", "632"]:
    dss.circuit.set_active_bus(bus)
    v = np.array(dss.bus.vmag_angle_pu)
    vpu = v[0::2] if v.size > 0 else []
    vmed = np.mean(vpu) if len(vpu) > 0 else np.nan
    status = "OK" if V_MIN <= vmed <= V_MAX else "FORA DO PRODIST"
    print(f"Barra {bus}: Vmed = {vmed:.4f} pu → {status}")



# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def get_all_buses(dss):
    buses = set() # guarda os nomes de todas os barramentos
    for ln in dss.lines.names: # retorna todas as linhas
        try:
            dss.circuit.set_active_element(f"Line.{ln}") # ativa como elemento atual
            for b in dss.cktelement.bus_names:
                buses.add(b.split('.')[0]) # rwemove o sufixo, fica so com o nome
        except Exception:
            pass
    for ld in dss.loads.names:
        try:
            dss.circuit.set_active_element(f"Load.{ld}")
            for b in dss.cktelement.bus_names:
                buses.add(b.split('.')[0])
        except Exception:
            pass
    for tf in dss.transformers.names:
        try:
            dss.circuit.set_active_element(f"Transformer.{tf}")
            for b in dss.cktelement.bus_names:
                buses.add(b.split('.')[0])
        except Exception:
            pass
    try:
        for cap in dss.capacitors.names:
            dss.circuit.set_active_element(f"Capacitor.{cap}")
            for b in dss.cktelement.bus_names:
                buses.add(b.split('.')[0])
    except Exception:
        pass
    return sorted(buses) # retorna a lista coms os barramentos ordenados

def safe_vmean(dss, bus): #tensão mèdia por fase (pu)
    try:
        dss.circuit.set_active_bus(bus) #ativa o barramento passado
        vvec = np.array(dss.bus.vmag_angle_pu) # vetor com as tensões (modulo e fase)
        if vvec is None or vvec.size == 0:
            return np.nan
        mags = vvec[0::2].astype(float) # pegas as amplitudes das tensões
        return float(np.nanmean(mags)) # retorna a tensão media do barramento
    except Exception:
        return np.nan

# -----------------------------
# LISTAS E INICIALIZAÇÕES
# -----------------------------
bus_names = get_all_buses(dss)
bus_names_master = bus_names[:]

load_names = list(dss.loads.names)
line_names = list(dss.lines.names)
transformer_names = list(dss.transformers.names)

# captura kW nominais das cargas (referência)
original_kw = {}
for load in load_names:
    try:
        dss.circuit.set_active_element(f"Load.{load}")
        pvec = dss.cktelement.powers
        if pvec is not None and len(pvec) > 0:
            original_kw[load] = float(np.nansum(pvec[0::2]))
        else:
            try:
                original_kw[load] = float(dss.loads.kw)
            except Exception:
                original_kw[load] = 0.0
    except Exception:
        original_kw[load] = 0.0

# -----------------------------
# CURVA DE CARGA (24h) - multiplicadores
# -----------------------------
loadshape = np.array([
    0.274841753, 0.22708204, 0.209259, 0.202058, 0.208986, 0.303014,
    0.573563, 0.591386, 0.455778, 0.364677, 0.333772, 0.292565,
    0.275164, 0.258334, 0.265285, 0.265905, 0.275512, 0.317736,
    0.519275, 0.815241, 0.841976, 0.771702, 0.578156, 0.378851
])

# cria loadshape no OpenDSS (nome: myshape)
mult_str = " ".join([str(float(x)) for x in loadshape])
dss.text(f"new loadshape.myshape npts=24 interval=1 mult=({mult_str})")

# associa daily=myshape a todas as loads (mantendo PF)
for load in load_names:
    try:
        dss.text(f"edit load.{load} daily=myshape")
    except Exception:
        pass

# configura solver para modo diário
dss.text("set mode=daily")
dss.text("set stepsize=1h")
dss.text("set number=24")

# -----------------------------
# containers de resultados
# -----------------------------
rows = []
tensoes_rows = []
perdas_linhas = []

# -----------------------------
# LOOP DIÁRIO (Time Series)
# -----------------------------
for hour in range(0, 24):
    # sincroniza hora do OpenDSS
    dss.text(f"set hour={hour}")
    dss.text("solve") # executa o fluxo de potência de cada hora

    # tensões por barramento
    for bus in bus_names: # percorre todos os barramentos
        vmean = safe_vmean(dss, bus) # vmean recebe a tensão media (pu) do barramento no instante atual
        try:
            dss.circuit.set_active_bus(bus)
            vvec = np.array(dss.bus.vmag_angle_pu) # tensão por fase
            if vvec is None or vvec.size == 0:
                mags = [np.nan, np.nan, np.nan]
            else:
                mags = vvec[0::2].astype(float).tolist()
                while len(mags) < 3:
                    mags.append(np.nan)
        except Exception:
            mags = [np.nan, np.nan, np.nan]

        tensoes_rows.append({ # salva os resultados
            "Hora_idx": hour,
            "Hora": hour + 1,
            "Barra": bus,
            "V_A_pu": mags[0],
            "V_B_pu": mags[1],
            "V_C_pu": mags[2],
            "V_mean_pu": vmean
        })

    # perdas por linha
    for line in line_names: # percorre todas as linhas
        try:
            dss.circuit.set_active_element(f"Line.{line}") # ativas todas as linhas
            pvec = np.array(dss.cktelement.powers) # coleta todas as potencias ativa
            losses_vec = dss.cktelement.losses # coleta todas as potencias reativa
        except Exception:
            pvec = np.array([])
            losses_vec = None

        Flow_A_kW = Flow_B_kW = Flow_C_kW = np.nan
        Flow_total_kW = np.nan
        Perda_A_kW = Perda_B_kW = Perda_C_kW = np.nan
        Perda_total_kW = np.nan
        Perda_line_kW = np.nan

        if pvec is not None and pvec.size >= 2:
            n = pvec.size
            n_phases = max(1, n // 4) if n >= 4 else 1
            try:
                P_from = pvec[0:2 * n_phases:2]
                P_to = pvec[2 * n_phases:4 * n_phases:2]
            except Exception:
                P_from = np.array([])
                P_to = np.array([])

            P_from = np.pad(P_from, (0, max(0, 3 - len(P_from))), constant_values=np.nan)
            P_to = np.pad(P_to, (0, max(0, 3 - len(P_to))), constant_values=np.nan)

            Flow_A_kW, Flow_B_kW, Flow_C_kW = P_from.tolist()
            Flow_total_kW = float(np.nansum(P_from))

            try:
                per_phase = (P_from + P_to).astype(float)
                Perda_A_kW, Perda_B_kW, Perda_C_kW = per_phase.tolist()
                Perda_total_kW = float(np.nansum(per_phase))
            except Exception:
                Perda_A_kW = Perda_B_kW = Perda_C_kW = np.nan
                Perda_total_kW = np.nan

        try:
            if losses_vec is not None:
                Perda_line_kW = float(losses_vec[0]) / 1000.0
        except Exception:
            Perda_line_kW = np.nan

        # repartir se necessário
        if np.isnan(Perda_total_kW) or (not np.isnan(Perda_line_kW) and abs(Perda_total_kW - Perda_line_kW) > 1e-3):
            abs_flows = np.abs(np.array([Flow_A_kW, Flow_B_kW, Flow_C_kW], dtype=float))
            sum_abs = np.nansum(abs_flows)
            if sum_abs > 0:
                ratio = abs_flows / sum_abs
                Perda_A_kW, Perda_B_kW, Perda_C_kW = (Perda_line_kW * ratio).tolist()
            else:
                Perda_A_kW, Perda_B_kW, Perda_C_kW = (np.nan, np.nan, np.nan)
            Perda_total_kW = Perda_line_kW

        perdas_linhas.append({
            "Hora_idx": hour,
            "Hora": hour + 1,          # EXIBIR 1..24
            "Linha": line,
            "Flow_A_kW": Flow_A_kW,
            "Flow_B_kW": Flow_B_kW,
            "Flow_C_kW": Flow_C_kW,
            "Flow_total_kW": Flow_total_kW,
            "Perda_A_kW": Perda_A_kW,
            "Perda_B_kW": Perda_B_kW,
            "Perda_C_kW": Perda_C_kW,
            "Perda_total_kW": Perda_total_kW,
            "Perda_line_kW": Perda_line_kW
        })

    # indicadores do sistema
    p_total = 0.0
    for lname in load_names:
        try:
            dss.circuit.set_active_element(f"Load.{lname}")
            pvec = dss.cktelement.powers
            if pvec is not None and len(pvec) > 0:
                p_total += float(np.nansum(pvec[0::2]))
        except Exception:
            pass

    try:
        losses_sys = dss.circuit.losses
        p_loss_kw = float(losses_sys[0]) / 1000.0
        q_loss_kvar = float(losses_sys[1]) / 1000.0
    except Exception:
        p_loss_kw = np.nan
        q_loss_kvar = np.nan

    rows.append({
        "Hora_idx": hour,
        "Hora": hour + 1,          # EXIBIR 1..24
        "Multiplicador": float(loadshape[hour]),
        "P_total_kW": p_total,
        "V671_pu": safe_vmean(dss, "671"),
        "V634_pu": safe_vmean(dss, "634"),
        "Perdas_kW": p_loss_kw,
        "Perdas_kvar": q_loss_kvar
    })

# -----------------------------
# CRIA DATAFRAMES
# -----------------------------
df = pd.DataFrame(rows)
df_tensoes = pd.DataFrame(tensoes_rows)
df_perdas_linhas = pd.DataFrame(perdas_linhas)

# Remover duplicatas (defensivo)
before_dup = len(df_perdas_linhas)
df_perdas_linhas = df_perdas_linhas.drop_duplicates(subset=['Hora_idx','Linha'], keep='first').reset_index(drop=True)
after_dup = len(df_perdas_linhas)
print(f"Duplicatas removidas (se houver): {before_dup - after_dup}")

# -----------------------------
# VALIDAÇÃO: soma por hora vs perdas do sistema
# -----------------------------
soma_por_hora = df_perdas_linhas.groupby('Hora')['Perda_line_kW'].sum().reset_index()
compar = soma_por_hora.merge(df[['Hora','Perdas_kW']], on='Hora', how='left')
compar['Diff_kW'] = compar['Perdas_kW'] - compar['Perda_line_kW']
compar['Ratio'] = compar.apply(lambda r: (r['Perda_line_kW'] / r['Perdas_kW']) if (pd.notna(r['Perdas_kW']) and r['Perdas_kW'] != 0) else np.nan, axis=1)

print("\nComparação perdas (linhas vs sistema) por hora:")
print(compar)
print("Resumo Ratio: min, mean, max =", compar['Ratio'].min(), compar['Ratio'].mean(), compar['Ratio'].max())

print("\nNaNs por coluna em df_perdas_linhas:\n", df_perdas_linhas.isna().sum())

# Energia diária (kWh, passo=1h)
energia_linhas = df_perdas_linhas['Perda_line_kW'].sum()
energia_sistema = df['Perdas_kW'].sum()
print(f"\nEnergia perdida (linhas) = {energia_linhas:.3f} kWh; energia perdida (sistema) = {energia_sistema:.3f} kWh")
print("Diferença diária (sistema - linhas) = ", energia_sistema - energia_linhas)

# -----------------------------
# SALVA RESULTADOS (EXCEL)
# -----------------------------
with pd.ExcelWriter(out_excel_final, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name='Resumo_Horario', index=False)
    df_perdas_linhas.to_excel(writer, sheet_name='Perda_Por_Linha', index=False)
    df_tensoes.to_excel(writer, sheet_name='Tensoes_Barras_x_Hora', index=False)

    # resumo PRODIST (por barramento) - usa Hora 1..24
    summary = []
    for bus in bus_names_master:
        sub = df_tensoes[df_tensoes['Barra'] == bus]
        if sub.empty:
            continue
        horas_fora = sub[(sub['V_mean_pu'] < V_MIN) | (sub['V_mean_pu'] > V_MAX)]
        pct_fora = len(horas_fora) / len(sub) * 100.0
        summary.append({
            "Barra": bus,
            "V_min_pu": sub['V_mean_pu'].min(),
            "V_max_pu": sub['V_mean_pu'].max(),
            "Horas_total": len(sub),
            "Horas_fora_PRODIST": len(horas_fora),
            "Pct_horas_fora_%": pct_fora
        })
    df_tensoes_summary = pd.DataFrame(summary)
    df_tensoes_summary.to_excel(writer, sheet_name='Resumo_PRODIST', index=False)

print(f"\nArquivo consolidado salvo em: {out_excel_final}")

# -----------------------------
# GRÁFICOS (EIXO 1..24)
# -----------------------------
# eixo para plot: Hora exibida 1..24
x_plot = df['Hora']  # 1..24
xticks = np.arange(1, 25, 2)  # 1,3,5,...,25? we'll cap at 24
xticks = xticks[xticks <= 24]
xlabels = [str(int(h)) for h in xticks]  # show 1..24

plt.figure(figsize=(9, 4))
plt.plot(x_plot, df["P_total_kW"], '-o')
plt.title("Curva de Carga do Sistema (Potência Total) — Caso 1")
plt.xlabel("Hora (1..24)")
plt.ylabel("P total (kW)")
plt.xticks(xticks, xlabels)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "curva_carga_1_24.png"))
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(x_plot, df["V671_pu"], '-o', label="Barra 671")
plt.plot(x_plot, df["V634_pu"], '-s', label="Barra 634")
plt.axhline(V_MAX, linestyle='--')
plt.axhline(V_MIN, linestyle='--')
plt.title("Tensão Horária (p.u.) — Barras 671 e 634")
plt.xlabel("Hora (1..24)")
plt.ylabel("Tensão (p.u.)")
plt.xticks(xticks, xlabels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "tensoes_671_634_1_24.png"))
plt.show()

plt.figure(figsize=(9, 4))
plt.plot(x_plot, df["Perdas_kW"], '-o', label='Perdas Totais (kW)')
# compar usa Hora (1..24)
plt.plot(compar['Hora'], compar['Perda_line_kW'], '-s', label='Soma Perdas Linhas (kW)')
plt.title("Perdas Totais vs Soma Perdas das Linhas")
plt.xlabel("Hora (1..24)")
plt.ylabel("Perdas (kW)")
plt.xticks(xticks, xlabels)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "perdas_vs_linhas_1_24.png"))
plt.show()

# ============================================
# GRÁFICO COMPARATIVO 1
# Multiplicador x P_total_kW
# ============================================
plt.figure(figsize=(10, 4))
plt.plot(df["Hora"], df["Multiplicador"], '-o', label="Multiplicador (Loadshape)")
plt.plot(df["Hora"], df["P_total_kW"] / df["P_total_kW"].max(), '-s', label="P_total_kW (normalizado)")
plt.title("Comparação: Multiplicador vs Potência Total do Sistema")
plt.xlabel("Hora")
plt.ylabel("Valor (normalizado)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================
# GRÁFICO COMPARATIVO 2
# Tensões das barras críticas (671 e 634)
# ============================================
plt.figure(figsize=(10, 4))
plt.plot(df["Hora"], df["V671_pu"], '-o', label="Tensão Barra 671")
plt.plot(df["Hora"], df["V634_pu"], '-s', label="Tensão Barra 634")
plt.axhline(0.93, linestyle='--', color='r', label="Limite PRODIST")
plt.axhline(1.05, linestyle='--', color='r')
plt.title("Tensão nas Barras 671 e 634 (24h)")
plt.xlabel("Hora")
plt.ylabel("Tensão (p.u.)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================
# GRÁFICO COMPARATIVO 3
# Potência total x Tensão média do sistema
# ============================================
df["V_media_sist"] = df_tensoes.groupby("Hora")["V_mean_pu"].mean().values

plt.figure(figsize=(10, 4))
plt.plot(df["Hora"], df["P_total_kW"], '-o', label="P_total_kW")
plt.plot(df["Hora"], df["V_media_sist"] * df["P_total_kW"].max(), '-s',
         label="Tensão Média x Escala da Potência")
plt.title("Relação Tensão Média da Rede vs Potência Entregue")
plt.xlabel("Hora")
plt.ylabel("Escala (kW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
# GRÁFICO: Tensões de todas as barras (Time Series)
# ============================================

plt.figure(figsize=(12, 6))

# colormap para as barras
cmap = plt.cm.tab20
colors = cmap(np.linspace(0, 1, len(bus_names_master)))

for i, bus in enumerate(bus_names_master):
    sub = df_tensoes[df_tensoes['Barra'] == bus]
    if sub.empty:
        continue
    plt.plot(
        sub['Hora'],
        sub['V_mean_pu'],
        label=f"Barra {bus}",
        color=colors[i],
        linewidth=1.5
    )

# Limites PRODIST
plt.axhline(V_MIN, linestyle='--', color='red', label='V_MIN PRODIST')
plt.axhline(V_MAX, linestyle='--', color='red', label='V_MAX PRODIST')

plt.title("Tensões por Barra (V_mean) — 24h")
plt.xlabel("Hora (1..24)")
plt.ylabel("Tensão (p.u.)")
plt.xticks(np.arange(1, 25, 1))
plt.grid(True)
plt.legend(title="Barramentos", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()

# salvar figura
plt.savefig(os.path.join(figures_dir, "tensoes_todas_barras_1_24.png"))
plt.show()

plt.figure(figsize=(10, 5))

for bus in bus_names_master:
    sub = df_tensoes[df_tensoes["Barra"] == bus]
    plt.plot(sub["Hora"], sub["V_mean_pu"], marker='o', linewidth=1, label=f"Barra {bus}")

plt.axhline(V_MIN, linestyle='--')
plt.axhline(V_MAX, linestyle='--')

plt.title("Tensões por Barra (V_mean) — 24h")
plt.xlabel("Hora")
plt.ylabel("Tensão (p.u.)")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

P_A = np.array(dss.monitors.channel(1))
P_B = np.array(dss.monitors.channel(2))
P_C = np.array(dss.monitors.channel(3))

P_total = P_A + P_B + P_C

plt.figure()
plt.plot(P_total)
plt.xlabel("Hora")
plt.ylabel("Potência Ativa Total (kW)")
plt.title("Potência Ativa na Subestação")
plt.grid(True)
plt.show()