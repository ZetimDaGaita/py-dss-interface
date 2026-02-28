import py_dss_interface
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Caminhos
# -----------------------------
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"
bus_coords = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Node_BusXY.csv"

# ----------------------------------------
# 1) Inicialização e compilação do circuito
# ----------------------------------------
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")

# ----------------------------------------
# 2) Loadshape residencial e time-series
# ----------------------------------------
loadshape = [
    0.274841753, 0.22708204, 0.209259, 0.202058, 0.208986, 0.303014,
    0.573563, 0.591386, 0.455778, 0.364677, 0.333772, 0.292565,
    0.275164, 0.258334, 0.265285, 0.265905, 0.275512, 0.317736,
    0.519275, 0.815241, 0.841976, 0.771702, 0.578156, 0.378851
]

dss.text("new loadshape.residencial npts=24 interval=1 mult=(" + " ".join(map(str, loadshape)) + ")")

# Aplicar o loadshape
for load in dss.loads.names:
    dss.circuit.set_active_element("load." + load)
    dss.loads.daily = "residencial"

# Configurar simulação time-series
dss.text("set mode=daily")
dss.text("set stepsize=1h")
dss.text("set number=24")
dss.text("solve")

# ----------------------------------------
# 3) Salvar tensões hora a hora
# ----------------------------------------
buses = dss.circuit.buses_names
hours = []
vmatrix = []

for h in range(24):
    vpu_hour = []
    for bus in buses:
        dss.circuit.set_active_bus(bus)
        vmag = np.array(dss.bus.vmag_angle_pu)
        vpu_hour.append(vmag.mean())
    vmatrix.append(vpu_hour)
    hours.append(h)
    dss.text("solve")  # avança uma hora

df_vhour = pd.DataFrame(vmatrix, index=hours, columns=buses)

# horas críticas
hora_pico = df_vhour.mean(axis=1).idxmin()
hora_vale = df_vhour.mean(axis=1).idxmax()

print("\nHora de pico =", hora_pico)
print("Hora de vale =", hora_vale)

# valores min/max para colorbar
vmin = df_vhour.min().min()
vmax = df_vhour.max().max()

# ----------------------------------------
# 4) Coordenadas das barras
# ----------------------------------------
coords = pd.read_csv(bus_coords, header=None, names=["Bus", "X", "Y"])
df = pd.DataFrame({"Bus": buses})
df = df.merge(coords, on="Bus", how="left")

# Criar barras artificiais
if "sourcebus" not in df["Bus"].values:
    df.loc[len(df)] = ["sourcebus", None, None]

if "rg60" not in df["Bus"].values:
    df.loc[len(df)] = ["rg60", None, None]

# sourcebus próximo a 650
x650, y650 = df.loc[df["Bus"]=="650", ["X","Y"]].values[0]
df.loc[df["Bus"]=="sourcebus", ["X","Y"]] = [x650 - 20, y650 + 10]

# rg60 entre 650 e 632
x632, y632 = df.loc[df["Bus"]=="632", ["X","Y"]].values[0]
xr = x650 + 0.30*(x632 - x650)
yr = y650 + 0.30*(y632 - y650)
df.loc[df["Bus"]=="rg60", ["X","Y"]] = [xr, yr]

# ----------------------------------------
# 5) Conexões da topologia
# ----------------------------------------
edges = []

# Linhas
for line in dss.lines.names:
    dss.circuit.set_active_element(f"line.{line}")
    b1, b2 = [b.split(".")[0] for b in dss.cktelement.bus_names]
    edges.append((b1, b2, "Line"))

# Transformadores
print("\nTransformadores:")
for trafo in dss.transformers.names:
    dss.circuit.set_active_element(f"transformer.{trafo}")
    b1, b2 = [b.split(".")[0] for b in dss.cktelement.bus_names]
    edges.append((b1, b2, "Transformer"))
    print(trafo, "→", dss.cktelement.bus_names)

# Capacitor
for cap in dss.capacitors.names:
    dss.circuit.set_active_element(f"capacitor.{cap}")
    b = dss.cktelement.bus_names[0].split(".")[0]
    edges.append((b, b, "Capacitor"))

# Cargas
for load in dss.loads.names:
    dss.circuit.set_active_element(f"load." + load)
    b = dss.cktelement.bus_names[0].split(".")[0]
    edges.append((b, b, "Load"))

# ----------------------------------------
# 6) Função genérica de plotagem
# ----------------------------------------
def plot_topologia(df, edges, titulo, vpu=None):
    plt.figure(figsize=(12, 9))
    offset_y = 8
    offset_x = 10

    # Desenhar conexões
    for b1, b2, etype in edges:
        if b1 in df["Bus"].values and b2 in df["Bus"].values:
            x1, y1 = df.loc[df["Bus"]==b1, ["X","Y"]].values[0]
            x2, y2 = df.loc[df["Bus"]==b2, ["X","Y"]].values[0]

            if etype == "Line":
                plt.plot([x1,x2], [y1,y2], "gray", lw=1)
            elif etype == "Transformer":
                plt.plot([x1,x2], [y1,y2], "b--", lw=1.5)
            elif etype == "Capacitor":
                plt.scatter(x1+offset_x, y1-offset_y, color="red", marker="s", s=60)
            elif etype == "Load":
                plt.scatter(x1, y1-offset_y, color="green", marker="^", s=60)

    # Barras
    if vpu is None:
        # SEM tensões (azul)
        plt.scatter(df["X"], df["Y"], color="blue", edgecolor="k", s=80)
    else:
        # COM tensões (gradiente)
        sc = plt.scatter(df["X"], df["Y"], c=vpu, cmap="coolwarm",
                         edgecolor="k", s=120, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, label="Tensão [p.u.]")

    # Rótulos
    for _, row in df.iterrows():
        plt.text(row["X"], row["Y"]+7, row["Bus"], fontsize=8, ha="center")

    plt.title(titulo)
    plt.xlabel("Coordenada X [m]")
    plt.ylabel("Coordenada Y [m]")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    plt.show()

# ----------------------------------------
# 7) Plotagens finais
# ----------------------------------------
plot_topologia(df, edges, "Topologia IEEE-13 — Sem Tensões")

plot_topologia(df, edges,
               f"Topologia — Hora de Vale (h = {hora_vale})",
               vpu=df_vhour.loc[hora_vale].values)

plot_topologia(df, edges,
               f"Topologia — Hora de Pico (h = {hora_pico})",
               vpu=df_vhour.loc[hora_pico].values)