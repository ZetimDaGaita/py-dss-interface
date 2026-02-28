import py_dss_interface
import pandas as pd

# Caminho do circuito
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"

# Inicialização
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")
dss.text("solve")

loads = dss.loads.names

# Função modelo final
def modelo_str(modelo_num, is_delta):
    modelo = {
        1: "PQ",
        2: "ConstPF",
        3: "Z",
        4: "I",
        5: "ConstI"
    }.get(modelo_num, f"Modelo{modelo_num}")
    return ("D" if is_delta else "Y") + "–" + modelo

# Tabela bruta
linhas = []

for load in loads:
    dss.loads.name = load

    bus_raw = dss.text(f"? load.{load}.bus1").strip()
    if bus_raw == "" or "Unknown" in bus_raw:
        continue

    partes = bus_raw.split(".")
    barra = partes[0]
    fases = list(map(int, partes[1:]))

    modelo = modelo_str(dss.loads.model, bool(dss.loads.is_delta))

    kw_total = dss.loads.kw
    kvar_total = dss.loads.kvar

    # Inicializar
    A = B = C = 0
    Ar = Br = Cr = 0

    if len(fases) == 1:
        f = fases[0]
        if f == 1: A, Ar = kw_total, kvar_total
        if f == 2: B, Br = kw_total, kvar_total
        if f == 3: C, Cr = kw_total, kvar_total

    elif len(fases) == 2:
        kw_f = kw_total/2
        kvar_f = kvar_total/2
        if 1 in fases: A, Ar = kw_f, kvar_f
        if 2 in fases: B, Br = kw_f, kvar_f
        if 3 in fases: C, Cr = kw_f, kvar_f

    elif len(fases) == 3:
        A = B = C = kw_total/3
        Ar = Br = Cr = kvar_total/3

    linhas.append([barra, modelo, A, Ar, B, Br, C, Cr])

# ============================================================
# AGRUPAR EM UMA LINHA POR BARRA
# ============================================================

df = pd.DataFrame(linhas, columns=[
    "Barra","Modelo","A_kW","A_kVAr","B_kW","B_kVAr","C_kW","C_kVAr"
])

# Se houver modelos diferentes na mesma barra → escreve "Misto"
def resolve_modelos(series):
    unicos = series.unique()
    return unicos[0] if len(unicos)==1 else "Misto"

df_final = df.groupby("Barra").agg({
    "Modelo": resolve_modelos,
    "A_kW": "sum",
    "A_kVAr": "sum",
    "B_kW": "sum",
    "B_kVAr": "sum",
    "C_kW": "sum",
    "C_kVAr": "sum"
}).reset_index()

print(df_final)
