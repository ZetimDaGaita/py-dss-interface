import py_dss_interface
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
circuito = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Nodeckt.dss"
bus_coords = r"C:\Users\josea\OneDrive\Desktop\Eng\TCC\OpenDSS\Meu_EX\Teste\13Bus\IEEE13Node_BusXY.csv"

# ----------------------------------------
# 1) Inicialização e compilação do circuito
# ----------------------------------------
dss = py_dss_interface.DSS()
dss.text(f"compile [{circuito}]")



#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# checar enabled/tap de cada regcontrol conhecido
for i in range(1,4):
    try:
        enabled = dss.text(f"? regcontrol.reg{i}.enabled")   # normalmente retorna 'true'/'false'
        tapnum = dss.text(f"? regcontrol.reg{i}.TapNum")
        print(f"reg{i}: enabled={enabled.strip()}, TapNum={tapnum.strip()}")
    except Exception as e:
        print(f"reg{i}: não encontrado ou erro -> {e}")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# agrupa perdas por hora
soma_por_hora = df_perdas_linhas.groupby('Hora')['Perda_line_kW'].sum().reset_index()
compar = soma_por_hora.merge(df[['Hora','Perdas_kW']], on='Hora', how='left')
compar['Diff_kW'] = compar['Perdas_kW'] - compar['Perda_line_kW']  # perdas sistema - somatorio linhas
compar['Ratio'] = compar['Perda_line_kW'] / compar['Perdas_kW']

print(compar)
print("Resumo Ratio: min, mean, max =", compar['Ratio'].min(), compar['Ratio'].mean(), compar['Ratio'].max())

# checa duplicatas Hora+Linha
dup = df_perdas_linhas.groupby(['Hora','Linha']).size().reset_index(name='count')
dups = dup[dup['count']>1]
print("Duplicatas (Hora,Linha) encontradas:", len(dups))
if not dups.empty:
    print(dups.head())

# checa NaNs
print("NaNs por coluna:\n", df_perdas_linhas.isna().sum())

# energia diária (kWh se passo=1h)
energia_linhas = df_perdas_linhas['Perda_line_kW'].sum()
energia_sistema = df['Perdas_kW'].sum()
print(f"Energia perdida (linhas) = {energia_linhas:.3f} kWh; energia perdida (sistema) = {energia_sistema:.3f} kWh")
print("Diferença diária (sistema - linhas) = ", energia_sistema - energia_linhas)