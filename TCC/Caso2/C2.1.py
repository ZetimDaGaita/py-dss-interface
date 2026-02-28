import numpy as np
import pandas as pd
import opendssdirect as dss

# =========================================================
# PARTE 0 – DADOS DE ENTRADA (JÁ EXISTENTES NO SEU SCRIPT)
# =========================================================

# Curva de carga (24 pontos)
loadshape = np.array(loadshape, dtype=float)

# Irradiância e temperatura (24 pontos)
pv_irradiance = np.array(pv_irradiance, dtype=float)
pv_temperature = np.array(pv_temperature, dtype=float)

# coeficiente térmico do PMPP
PMPP_TEMP_COEFF = -0.0045

# caminho do circuito
circuito = r"IEEE13Nodeckt.dss"   # ajuste se necessário

# =========================================================
# PARTE 1 – LIMPA E COMPILA CIRCUITO
# =========================================================

dss.text("Clear")
dss.text(f"Compile [{circuito}]")

# desativa controles automáticos
dss.text("set controlmode=off")

for i in range(1, 6):
    try:
        dss.text(f"disable regcontrol.reg{i}")
    except:
        pass

# =========================================================
# PARTE 2 – LOADSHAPE (CARGAS)
# =========================================================

mult_str = " ".join([f"{x:.6f}" for x in loadshape])

dss.text(
    f"New LoadShape.MyLoadShape npts=24 interval=1 mult=({mult_str})"
)

# associa LoadShape a TODAS as cargas
for ld in dss.loads.names:
    dss.text(f"Edit Load.{ld} daily=MyLoadShape")

# =========================================================
# PARTE 3 – TSHAPE + XYCURVE (PV)
# =========================================================

# curva de temperatura
temp_str = " ".join([f"{x:.3f}" for x in pv_temperature])
dss.text(
    f"New TShape.MyTshape npts=24 interval=1 temp=({temp_str})"
)

# curva P vs T
dss.text(
    "New XYCurve.PvsT npts=3 "
    "xarray=(0 25 75) "
    "yarray=(1.0 1.0 0.8)"
)

# curva de irradiância
irr_str = " ".join([f"{x:.6f}" for x in pv_irradiance])
dss.text(
    f"New LoadShape.MyIrrShape npts=24 interval=1 mult=({irr_str})"
)

# aplica aos PVs
dss.text(
    "Edit PVSystem.PV671 daily=MyIrrShape "
    "Tdaily=MyTshape P-TCurve=PvsT"
)

dss.text(
    "Edit PVSystem.PV634 daily=MyIrrShape "
    "Tdaily=MyTshape P-TCurve=PvsT"
)

# =========================================================
# PARTE 4 – MONITORES (CRIAÇÃO CORRETA)
# =========================================================

# Tensões (via elementos, não bus!)
dss.text("New Monitor.V_671 element=PVSystem.PV671 terminal=1 mode=0")
dss.text("New Monitor.V_634 element=PVSystem.PV634 terminal=1 mode=0")

# Potência dos PVs
dss.text("New Monitor.P_PV671 element=PVSystem.PV671 terminal=1 mode=1")
dss.text("New Monitor.P_PV634 element=PVSystem.PV634 terminal=1 mode=1")

# perdas do sistema (linha da subestação)
dss.text("New Monitor.LossSys element=Line.650632 terminal=1 mode=9")

# =========================================================
# PARTE 5 – SIMULAÇÃO DIÁRIA
# =========================================================

dss.text("set mode=daily")
dss.text("set stepsize=1h")
dss.text("set number=24")

dss.text("solve")

# =========================================================
# PARTE 6 – EXPORTAÇÃO
# =========================================================

print("Monitores ativos:", dss.monitors.names)

dss.text("Export Monitors")

# =========================================================
# FIM DO SCRIPT
# =========================================================
