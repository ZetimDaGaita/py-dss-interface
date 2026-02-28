
import py_dss_interface
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"

# Pasta para salvar as planilhas do Caso 2
excel_dir = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\Caso_2\Excel"
os.makedirs(excel_dir, exist_ok=True)

# Caminhos de todos os arquivos Excel
excel_final = os.path.join(excel_dir, "Resultados_Caso2_2SFV_400kW_model2.xlsx")
excel_base = os.path.join(excel_dir, "Resultados_Caso2_2SFV_400kW.xlsx")
excel_pvst = os.path.join(excel_dir, "Resultados_Caso2_2SFV_400kW_PvsT.xlsx")

# Pasta dos gráficos do Caso 2
out_plots_dir = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\Caso_2\Graficos"
os.makedirs(out_plots_dir, exist_ok=True)

V_MIN, V_MAX = 0.93, 1.05  # PRODIST (p.u.)

PV_POWER_671 = 400.0  # kW (Pmpp)
PV_POWER_634 = 400.0  # kW (Pmpp)

KV_671 = 4.16 / math.sqrt(3)
KV_634 = 0.48 / math.sqrt(3)

# coeficiente linear para gerar a curva PvsT (caso não exista curva do fabricante)
PMPP_TEMP_COEFF = -0.004  # -0.4 %/°C



# -----------------------------
# CURVAS HORÁRIAS (24 horas)
# -----------------------------
loadshape = np.array([
    0.274841753, 0.22708204, 0.209259, 0.202058, 0.208986, 0.303014,
    0.573563, 0.591386, 0.455778, 0.364677, 0.333772, 0.292565,
    0.275164, 0.258334, 0.265285, 0.265905, 0.275512, 0.317736,
    0.519275, 0.815241, 0.841976, 0.771702, 0.578156, 0.378851
])

pv_irradiance = np.array([
    0.0, 0.0, 0.0, 0.0, 0.01, 0.05,
    0.30, 0.55, 0.75, 0.85, 0.95, 0.98,
    1.00, 0.95, 0.85, 0.70, 0.50, 0.30,
    0.10, 0.02, 0.0, 0.0, 0.0, 0.0
])

pv_temperature = np.array([
    20, 19, 18, 17, 18, 20,
    25, 28, 32, 34, 35, 36,
    37, 36, 34, 32, 30, 28,
    25, 22, 20, 19, 18, 18
])

# -----------------------------
# Inicializa OpenDSS
# -----------------------------
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")

# desativa controladores/reguladores
for i in range(1, 6):
    try:
        dss.text(f"disable regcontrol.reg{i}")
    except:
        pass
dss.text("set controlmode=off")

# -----------------------------
# Criar XYCurve PvsT
# -----------------------------
pvsT_name = "PvsT_24"
temps_list = pv_temperature.tolist()

pvsT_y = [1.0 + (PMPP_TEMP_COEFF * (T - 25)) for T in temps_list]

xarray_str = " ".join([str(int(t)) for t in temps_list])
yarray_str = " ".join([f"{y:.6f}" for y in pvsT_y])

dss.text(f"New XYCurve.{pvsT_name} npts=24 xarray=[{xarray_str}] yarray=[{yarray_str}]")

# -----------------------------
# Criar XYCurve de eficiência do inversor
# -----------------------------
eff_name = "InvEff_4pt"
eff_x = [0.0, 0.2, 0.5, 1.0]
eff_y = [0.90, 0.95, 0.97, 0.98]
dss.text(f"New XYCurve.{eff_name} npts=4 xarray=[0 0.2 0.5 1.0] yarray=[0.90 0.95 0.97 0.98]")

# -----------------------------
# Criar PVSystem model=2
# -----------------------------
def ensure_pv_model2(name, bus, kv, pmpp_kw, phases=3, kVA=None):
    existing = []
    try:
        existing = list(dss.pvsystems.names)
    except:
        existing = []

    kvastr = int(kVA) if kVA is not None else int(pmpp_kw)

    if name.lower() in [n.lower() for n in existing]:
        dss.text(f"Edit PVSystem.{name} kV={kv:.6f} kVA={kvastr} "
                 f"Pmpp={pmpp_kw} model=2 P-TCurve={pvsT_name} effcurve={eff_name}")
        return

    dss.text(
        f"New PVSystem.{name} phases={phases} bus1={bus} kV={kv:.6f} "
        f"kVA={kvastr} Pmpp={pmpp_kw} Irradiance=0.0 model=2 "
        f"P-TCurve={pvsT_name} effcurve={eff_name} pf=1 conn=wye enabled=yes"
    )

ensure_pv_model2("PV671", "671", KV_671, PV_POWER_671)
ensure_pv_model2("PV634", "634", KV_634, PV_POWER_634)

dss.text("solve")

# -----------------------------
# PARTE 2 - Funções auxiliares e inicializações
# -----------------------------

def get_all_buses(dss):
    buses = set()
    try:
        for ln in dss.lines.names:
            try:
                dss.circuit.set_active_element(f"Line.{ln}")
                for b in dss.cktelement.bus_names:
                    buses.add(b.split('.')[0])
            except:
                pass
    except:
        pass
    try:
        for ld in dss.loads.names:
            try:
                dss.circuit.set_active_element(f"Load.{ld}")
                for b in dss.cktelement.bus_names:
                    buses.add(b.split('.')[0])
            except:
                pass
    except:
        pass
    try:
        for tf in dss.transformers.names:
            try:
                dss.circuit.set_active_element(f"Transformer.{tf}")
                for b in dss.cktelement.bus_names:
                    buses.add(b.split('.')[0])
            except:
                pass
    except:
        pass
    try:
        for cap in dss.capacitors.names:
            try:
                dss.circuit.set_active_element(f"Capacitor.{cap}")
                for b in dss.cktelement.bus_names:
                    buses.add(b.split('.')[0])
            except:
                pass
    except:
        pass
    return sorted(buses)


def safe_vmean(dss, bus):
    try:
        dss.circuit.set_active_bus(bus)
        vvec = np.array(dss.bus.vmag_angle_pu)
        if vvec is None or vvec.size == 0:
            return np.nan
        mags = vvec[0::2].astype(float)
        return float(np.nanmean(mags))
    except:
        return np.nan


# captura nomes e listas do circuito
bus_names = get_all_buses(dss)
load_names = list(dss.loads.names)
line_names = list(dss.lines.names)
transformer_names = list(dss.transformers.names)

# captura kW originais das cargas (base)
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
            except:
                original_kw[load] = 0.0
    except:
        original_kw[load] = 0.0

# containers para resultados e séries de plots
tensoes_rows = []
perdas_linhas = []
rows = []

hrs = list(range(24))
irr_series = []
temp_series = []
pvsT_factor_series = []

pv671_set = []
pv634_set = []
pv671_python_corr = []
pv634_python_corr = []
pv671_opendss = []
pv634_opendss = []

v671_series = []
v634_series = []
losses_sys_series = []
losses_lines_sum_series = []

# Preparado para o loop horário (Parte 3 seguirá com o loop)
print("Parte 2 carregada: funções auxiliares, listas e containers inicializados.")

# -----------------------------
# PARTE 3 - Loop horário (24 horas)
# -----------------------------
mult_str = " ".join([str(float(x)) for x in loadshape])
dss.text(f"New LoadShape.myshape npts=24 interval=1 mult=({mult_str})")

for load in load_names:
    dss.text(f"Edit Load.{load} daily=myshape")

    dss.text("set mode=daily")
    dss.text("set stepsize=1h")
    dss.text("set number=24")

    # -----------------------------
    # Irradiância e temperatura PV
    # -----------------------------

    irr_str = " ".join([str(float(x)) for x in pv_irradiance])
    dss.text(f"New LoadShape.PV_Irr npts=24 interval=1 mult=({irr_str})")

    temp_str = " ".join([str(float(x)) for x in pv_temperature])
    dss.text(f"New Tshape.PV_Temp npts=24 interval=1 temp=({temp_str})")

    dss.text("Edit PVSystem.PV671 daily=PV_Irr Tdaily=PV_Temp")
    dss.text("Edit PVSystem.PV634 daily=PV_Irr Tdaily=PV_Temp")

    dss.text("solve")

    # -----------------------------
    # Tensões por barra
    # -----------------------------
    for bus in bus_names:
        vmean = safe_vmean(dss, bus)
        try:
            dss.circuit.set_active_bus(bus)
            vvec = np.array(dss.bus.vmag_angle_pu)
            if vvec is None or vvec.size == 0:
                mags = [np.nan, np.nan, np.nan]
            else:
                mags = vvec[0::2].astype(float).tolist()
                while len(mags) < 3:
                    mags.append(np.nan)
        except:
            mags = [np.nan, np.nan, np.nan]

        tensoes_rows.append({
            "Hora": h,
            "Barra": bus,
            "V_A_pu": mags[0],
            "V_B_pu": mags[1],
            "V_C_pu": mags[2],
            "V_mean_pu": vmean
        })

    # -----------------------------
    # Perdas por linha
    # -----------------------------
    for line in line_names:
        try:
            dss.circuit.set_active_element(f"Line.{line}")
            pvec = np.array(dss.cktelement.powers)
            losses_vec = dss.cktelement.losses
        except:
            pvec = np.array([])
            losses_vec = None

        Flow_A_kW = Flow_B_kW = Flow_C_kW = np.nan
        Flow_total_kW = np.nan
        Perda_A_kW = Perda_B_kW = Perda_C_kW = np.nan
        Perda_total_kW = np.nan
        Perda_line_kW = np.nan

        if pvec is not None and pvec.size >= 2:
            n = pvec.size
            n_ph = max(1, n // 4) if n >= 4 else 1
            try:
                P_from = pvec[0:2*n_ph:2]
                P_to = pvec[2*n_ph:4*n_ph:2]
            except:
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
            except:
                pass

        try:
            if losses_vec is not None:
                Perda_line_kW = float(losses_vec[0]) / 1000.0
        except:
            Perda_line_kW = np.nan

        # se tiver divergência, recalcula proporcionalmente
        if np.isnan(Perda_total_kW) or (not np.isnan(Perda_line_kW)
                                        and abs(Perda_total_kW - Perda_line_kW) > 1e-3):

            abs_fl = np.abs(np.array([Flow_A_kW, Flow_B_kW, Flow_C_kW], dtype=float))
            s_abs = np.nansum(abs_fl)
            if s_abs > 0:
                ratio = abs_fl / s_abs
                Perda_A_kW, Perda_B_kW, Perda_C_kW = (Perda_line_kW * ratio).tolist()
            else:
                Perda_A_kW = Perda_B_kW = Perda_C_kW = np.nan

            Perda_total_kW = Perda_line_kW

        perdas_linhas.append({
            "Hora": h,
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

    # -----------------------------
    # Potência total das cargas
    # -----------------------------
    p_total_loads = 0.0
    for lname in load_names:
        try:
            dss.circuit.set_active_element(f"Load.{lname}")
            p_vec = dss.cktelement.powers
            if p_vec is not None and len(p_vec) > 0:
                p_total_loads += float(np.nansum(p_vec[0::2]))
        except:
            pass

    # -----------------------------
    # Potências definidas (set) e corrigidas (python)
    # -----------------------------
    pv671_set_kw = PV_POWER_671 * irr
    pv634_set_kw = PV_POWER_634 * irr

    pv671_corr_py = PV_POWER_671 * irr * f_temp
    pv634_corr_py = PV_POWER_634 * irr * f_temp

    # -----------------------------
    # Medição OpenDSS das potências
    # -----------------------------
    try:
        dss.circuit.set_active_element("PVSystem.PV671")
        pvec = dss.cktelement.powers
        if pvec is not None and len(pvec) > 0:
            pv671_meas = -float(np.nansum(np.array(pvec[0::2])))
        else:
            pv671_meas = pv671_corr_py
    except:
        pv671_meas = pv671_corr_py

    try:
        dss.circuit.set_active_element("PVSystem.PV634")
        pvec = dss.cktelement.powers
        if pvec is not None and len(pvec) > 0:
            pv634_meas = -float(np.nansum(np.array(pvec[0::2])))
        else:
            pv634_meas = pv634_corr_py
    except:
        pv634_meas = pv634_corr_py

    p_total_net = p_total_loads - (pv671_meas + pv634_meas)

    # -----------------------------
    # Tensões 671 e 634
    # -----------------------------
    v671 = safe_vmean(dss, "671")
    v634 = safe_vmean(dss, "634")

    # -----------------------------
    # Perdas totais do sistema
    # -----------------------------
    try:
        ls = dss.circuit.losses
        p_loss_kw = float(ls[0]) / 1000.0
        q_loss_kvar = float(ls[1]) / 1000.0
    except:
        p_loss_kw = np.nan
        q_loss_kvar = np.nan

    # soma das perdas das linhas naquela hora
    perda_linhas_h = np.nansum([e["Perda_line_kW"] for e in perdas_linhas if e["Hora"] == h])

    # -----------------------------
    # Salva linha do resumo horário
    # -----------------------------
    rows.append({
        "Hora": h,
        "Multiplicador": mult,
        "P_total_loads_kW": p_total_loads,
        "PV671_kW_set": pv671_set_kw,
        "PV634_kW_set": pv634_set_kw,
        "PV671_kW_py_corr": pv671_corr_py,
        "PV634_kW_py_corr": pv634_corr_py,
        "PV671_kW_meas": pv671_meas,
        "PV634_kW_meas": pv634_meas,
        "P_total_net_kW": p_total_net,
        "V671_pu": v671,
        "V634_pu": v634,
        "Perdas_kW": p_loss_kw,
        "Perdas_kvar": q_loss_kvar,
        "Temp_C": temp,
        "Irradiance_pu": irr,
        "PvsT_factor": f_temp,
        "Perda_linhas_sum_kW": perda_linhas_h
    })

    # -----------------------------
    # Alimenta séries para gráficos
    # -----------------------------
    irr_series.append(irr)
    temp_series.append(temp)
    pvsT_factor_series.append(f_temp)

    pv671_set.append(pv671_set_kw)
    pv634_set.append(pv634_set_kw)

    pv671_python_corr.append(pv671_corr_py)
    pv634_python_corr.append(pv634_corr_py)

    pv671_opendss.append(pv671_meas)
    pv634_opendss.append(pv634_meas)

    v671_series.append(v671)
    v634_series.append(v634)

    losses_sys_series.append(p_loss_kw)
    losses_lines_sum_series.append(perda_linhas_h)

print("Loop horário concluído.")

# -----------------------------
# PARTE 4 - DataFrames, consolidação e PRODIST
# -----------------------------

# Converte listas em DataFrames
df = pd.DataFrame(rows)
df_tensoes = pd.DataFrame(tensoes_rows)
df_perdas_linhas = pd.DataFrame(perdas_linhas)

# Remove duplicatas (por segurança)
df_perdas_linhas = df_perdas_linhas.drop_duplicates(
    subset=['Hora', 'Linha'], keep='first'
).reset_index(drop=True)

# Comparação perdas: soma linhas vs perdas do sistema
soma_por_hora = df_perdas_linhas.groupby('Hora')['Perda_line_kW'].sum().reset_index()
compar = soma_por_hora.merge(df[['Hora', 'Perdas_kW']], on='Hora', how='left')
compar['Diff_kW'] = compar['Perdas_kW'] - compar['Perda_line_kW']
compar['Ratio'] = compar.apply(
    lambda r: r['Perda_line_kW'] / r['Perdas_kW']
    if (not pd.isna(r['Perdas_kW']) and abs(r['Perdas_kW']) > 1e-9)
    else np.nan,
    axis=1
)

# Criação da pasta do Excel (já existe pela PARTE 1, mas garantimos)
os.makedirs(excel_dir, exist_ok=True)

# -------------------------------------
# SALVA PLANILHAS NAS PASTAS CORRETAS
# -------------------------------------

# 1) Primeiro arquivo base (sem PvsT)
with pd.ExcelWriter(excel_base, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name='Resumo_Horario', index=False)
    df_perdas_linhas.to_excel(writer, sheet_name='Perda_Por_Linha', index=False)
    df_tensoes.to_excel(writer, sheet_name='Tensoes_Barras_x_Hora', index=False)

# 2) Segundo arquivo PvsT (caso necessário)
with pd.ExcelWriter(excel_pvst, engine='openpyxl', mode='w') as writer:
    df[['Hora','Temp_C','Irradiance_pu','PvsT_factor']].to_excel(
        writer, sheet_name='PvsT', index=False
    )

# --------------------------------------------------
# ARQUIVO CONSOLIDADO FINAL (excel_final)
# --------------------------------------------------

with pd.ExcelWriter(excel_final, engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, sheet_name='Resumo_Horario', index=False)
    df_perdas_linhas.to_excel(writer, sheet_name='Perda_Por_Linha', index=False)
    df_tensoes.to_excel(writer, sheet_name='Tensoes_Barras_x_Hora', index=False)
    compar.to_excel(writer, sheet_name='Comparacao_Perdas', index=False)

# -----------------------------
# PRODIST — resumo tensão por barra
# -----------------------------
summary = []
for bus in bus_names:
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

# adiciona ao arquivo final
with pd.ExcelWriter(excel_final, engine='openpyxl', mode='a') as writer:
    df_tensoes_summary.to_excel(writer, sheet_name='Resumo_PRODIST', index=False)

print("\n📁 Arquivo consolidado salvo em:", excel_final)
print("📁 Arquivo base salvo em:", excel_base)
print("📁 Arquivo PvsT salvo em:", excel_pvst)

