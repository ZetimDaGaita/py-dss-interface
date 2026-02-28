import py_dss_interface
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"
out_excel = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\Resultados_Caso1_Simplificado.xlsx"
V_MIN, V_MAX = 0.93, 1.05

# -----------------------------
# INICIALIZAÇÃO
# -----------------------------
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")

# -----------------------------
# CRIAÇÃO AUTOMÁTICA DO LOADSHAPE IEEE (24 h)
# -----------------------------
multiplicadores_ieee = [
    0.274841753, 0.22708204, 0.209259, 0.202058, 0.208986, 0.303014,
    0.573563, 0.591386, 0.455778, 0.364677, 0.333772, 0.292565,
    0.275164, 0.258334, 0.265285, 0.265905, 0.275512, 0.317736,
    0.519275, 0.815241, 0.841976, 0.771702, 0.578156, 0.378851
]

loadshape_str = " ".join(str(v) for v in multiplicadores_ieee)

dss.text(f"new Loadshape.LoadshapeIEEE npts=24 interval=1 mult=({loadshape_str})")

# ASSOCIA A CURVA A TODAS AS CARGAS
for ld in dss.loads.names:
    dss.text(f"edit Load.{ld} yearly=LoadshapeIEEE")

# -----------------------------
# DESATIVAR REGULADORES
# -----------------------------
for i in range(1, 4):
    dss.text(f"disable regcontrol.reg{i}")
    dss.text(f"edit transformer.reg{i} taps=[1.0 1.0]")
dss.text("set controlmode=off")

# -----------------------------
# FUNÇÕES AUXILIARES
# -----------------------------
def mean_v(dss, bus):
    try:
        dss.circuit.set_active_bus(bus)
        v = np.array(dss.bus.vmag_angle_pu)
        return np.nanmean(v[0::2])
    except:
        return np.nan

barras = list(dss.circuit.buses_names)
linhas = list(dss.lines.names)

tensoes = []
perdas_linhas = []
resumo = []

# -----------------------------
# LOOP HORÁRIO (24 h)
# -----------------------------
for h in range(1, 25):
    dss.text(f"set hour={h}")
    dss.text("solve")

    # Tensões
    for b in barras:
        dss.circuit.set_active_bus(b)
        v = np.array(dss.bus.vmag_angle_pu)

        if v.size == 0:
            mags = [np.nan, np.nan, np.nan]
        else:
            mags = v[0::2].tolist()
            while len(mags) < 3:
                mags.append(np.nan)

        tensoes.append({
            "Hora": h,
            "Barra": b,
            "V_mean_pu": np.nanmean(mags)
        })

    # Perdas por linha
    for ln in linhas:
        dss.circuit.set_active_element(f"Line.{ln}")
        try:
            losses = dss.cktelement.losses[0] / 1000
        except:
            losses = np.nan

        perdas_linhas.append({
            "Hora": h,
            "Linha": ln,
            "Perda_kW": losses
        })

    # Resumo
    try:
        p_loss = dss.circuit.losses[0] / 1000
    except:
        p_loss = np.nan

    resumo.append({
        "Hora": h,
        "Perdas_kW": p_loss,
        "V634_pu": mean_v(dss, "634"),
        "V671_pu": mean_v(dss, "671")
    })

# -----------------------------
# DATAFRAMES
# -----------------------------
df_resumo = pd.DataFrame(resumo)
df_tensoes = pd.DataFrame(tensoes)
df_linhas = pd.DataFrame(perdas_linhas)

# -----------------------------
# PRODIST
# -----------------------------
summary = []
for b in barras:
    sub = df_tensoes[df_tensoes["Barra"] == b]
    hrs_out = sub[(sub["V_mean_pu"] < V_MIN) | (sub["V_mean_pu"] > V_MAX)]
    summary.append({
        "Barra": b,
        "Vmin": sub["V_mean_pu"].min(),
        "Vmax": sub["V_mean_pu"].max(),
        "Horas_fora": len(hrs_out),
        "Perc_fora_%": len(hrs_out) / len(sub) * 100
    })
df_prodist = pd.DataFrame(summary)

# -----------------------------
# SALVAR EXCEL
# -----------------------------
with pd.ExcelWriter(out_excel, engine='openpyxl') as w:
    df_resumo.to_excel(w, "Resumo", index=False)
    df_tensoes.to_excel(w, "Tensoes", index=False)
    df_linhas.to_excel(w, "Perdas_Linhas", index=False)
    df_prodist.to_excel(w, "PRODIST", index=False)

print(f"\nArquivo salvo em: {out_excel}")

# -----------------------------
# GRÁFICOS
# -----------------------------

# 1) Perdas totais
plt.figure(figsize=(8,4))
plt.plot(df_resumo["Hora"], df_resumo["Perdas_kW"], '-o')
plt.title("Perdas Totais (kW)")
plt.grid()
plt.show()

# 2) Tensões 634 e 671
plt.figure(figsize=(8,4))
plt.plot(df_resumo["Hora"], df_resumo["V634_pu"], label="634")
plt.plot(df_resumo["Hora"], df_resumo["V671_pu"], label="671")
plt.axhline(V_MIN, linestyle="--")
plt.axhline(V_MAX, linestyle="--")
plt.legend()
plt.grid()
plt.title("Tensões nas Barras 634 e 671")
plt.show()

# 3) TODAS AS BARRAS
plt.figure(figsize=(10,6))
for b in barras:
    sub = df_tensoes[df_tensoes["Barra"] == b]
    plt.plot(sub["Hora"], sub["V_mean_pu"], label=b)

plt.axhline(V_MIN, linestyle="--", color="red")
plt.axhline(V_MAX, linestyle="--", color="red")
plt.title("Tensões nos Barramentos - Todas as Barras")
plt.xlabel("Hora")
plt.ylabel("Tensão (p.u.)")
plt.grid()
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

df_resumo.to_excel(excel_writer=w, sheet_name="Resumo", index=False)
df_tensoes.to_excel(excel_writer=w, sheet_name="Tensoes", index=False)
df_linhas.to_excel(excel_writer=w, sheet_name="Perdas_Linhas", index=False)
df_prodist.to_excel(excel_writer=w, sheet_name="PRODIST", index=False)
