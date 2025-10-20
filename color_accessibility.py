# color_accessibility.py
# ---------------------------------------------------------
# Utilidades de paletas y filtros de daltonismo para Plotly
# Trabaja SIEMPRE en RGB lineal (sRGB desgamma/regamma)
# ---------------------------------------------------------

from __future__ import annotations
from typing import List, Literal, Dict, Any, Tuple, Union
import math

Mode = Literal['none', 'protan', 'deutan', 'tritan', 'achroma']

# ---- sRGB <-> Linear sRGB ----------------------------------------------------

def _srgb_to_linear(c: float) -> float:
    # c in [0,1] sRGB gamma
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _linear_to_srgb(c: float) -> float:
    # c in [0,1] linear
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0/2.4)) - 0.055

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# ---- Matrices de simulación daltonismo (Machado et al., aproximadas) --------
# Matrices para sRGB lineal (3x3). Son aproximaciones ampliamente usadas.
M_PROTAN = (
    (0.152286, 1.052583, -0.204868),
    (0.114503, 0.786281,  0.099216),
    (-0.003882, -0.048116, 1.051998),
)

M_DEUTAN = (
    (0.367322, 0.860646, -0.227968),
    (0.280085, 0.672501,  0.047413),
    (-0.011820, 0.042940, 0.968881),
)

M_TRITAN = (
    (1.255528, -0.076749, -0.178779),
    (-0.078411, 0.930809,  0.147602),
    (0.004733,  0.691367,  0.303900),
)

def _apply_matrix_lin(rgb_lin: Tuple[float, float, float], M: Tuple[Tuple[float,float,float], ...]) -> Tuple[float, float, float]:
    r, g, b = rgb_lin
    r2 = M[0][0]*r + M[0][1]*g + M[0][2]*b
    g2 = M[1][0]*r + M[1][1]*g + M[1][2]*b
    b2 = M[2][0]*r + M[2][1]*g + M[2][2]*b
    return (r2, g2, b2)

def _to_gray_lin(rgb_lin: Tuple[float,float,float]) -> Tuple[float,float,float]:
    # Luminancia BT.709
    r, g, b = rgb_lin
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    return (Y, Y, Y)

# ---- Paletas base ------------------------------------------------------------

from math import floor

def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a)*t

def _hex(r: float, g: float, b: float) -> str:
    R = int(round(_clamp01(r)*255))
    G = int(round(_clamp01(g)*255))
    B = int(round(_clamp01(b)*255))
    return f"#{R:02X}{G:02X}{B:02X}"

# (Se mantienen funciones viridis/cividis/plasma por compatibilidad interna — no usadas)
def _viridis_rgb(t: float) -> Tuple[float,float,float]:
    x = t
    r = 0.2803 + 0.2339*x + 0.0431*x*x - 0.3053*x**3 + 0.3341*x**4
    g = 0.0014 + 1.1110*x - 1.4816*x**2 + 1.1645*x**3 - 0.2934*x**4
    b = 0.2931 + 0.9679*x - 1.7622*x**2 + 1.1759*x**3 - 0.3087*x**4
    return (_clamp01(r), _clamp01(g), _clamp01(b))

def _cividis_rgb(t: float) -> Tuple[float,float,float]:
    x=t
    r = 0.0000 + 1.0190*x - 0.7320*x**2 + 0.1960*x**3
    g = 0.1375 + 0.8500*x + 0.0725*x**2 - 0.0600*x**3
    b = 0.3000 + 0.2900*x + 0.0500*x**2
    return (_clamp01(r), _clamp01(g), _clamp01(b))

def _plasma_rgb(t: float) -> Tuple[float,float,float]:
    x=t
    r = 0.051 + 2.588*x - 5.112*x**2 + 4.047*x**3 - 1.074*x**4
    g = 0.031 + 0.868*x + 0.465*x**2 - 0.921*x**3 + 0.458*x**4
    b = 0.532 + 0.133*x + 0.165*x**2 - 0.503*x**3 + 0.232*x**4
    return (_clamp01(r), _clamp01(g), _clamp01(b))

def _sample_base_palette(base: Union[str, List[str]], n:int=256) -> List[str]:
    """
    Devuelve una paleta de n colores en HEX.
    - Si base es lista HEX, la re-muestrea a n.
    - Paletas del proyecto:
        'pvout'      -> (imagen 2) azul→turquesa→verde→amarillo→naranja→rojo→púrpura
        'solar_day'  -> (imagen 1) verde→amarillo→naranja→rojo
    - Fallback: 'pvout'
    """

    # Si ya viene lista de hex -> upsample a n (en sRGB)
    if isinstance(base, list) and base and isinstance(base[0], str) and base[0].startswith('#'):
        m = len(base)
        if m == n:
            return base[:]
        out=[]
        def hex_to_rgb(h:str)->Tuple[float,float,float]:
            h=h.lstrip('#')
            return (int(h[0:2],16)/255.0, int(h[2:4],16)/255.0, int(h[4:6],16)/255.0)
        for i in range(n):
            u = i/(n-1)
            pos = u*(m-1)
            j = int(math.floor(pos))
            k = min(j+1, m-1)
            t = pos - j
            r1,g1,b1 = hex_to_rgb(base[j]); r2,g2,b2 = hex_to_rgb(base[k])
            r=_interp(r1,r2,t); g=_interp(g1,g2,t); b=_interp(b1,b2,t)
            out.append(_hex(r,g,b))
        return out

    # Paletas personalizadas extraídas de tus imágenes
    name = (base or 'pvout').lower()

    # Imagen 2 (PVOUT)
    pvout_seq = [
        "#2B3B73","#2F5C97","#367CAD","#3F9CC0","#54B9BE","#73CE9E","#9BD57C","#C7D96C",
        "#E9D764","#F8CF61","#FFC258","#FFA64A","#FF8A3F","#F66F3F","#E35744","#C6484F",
        "#A13F66","#7A3676","#532D6E","#2E2150"
    ]

    # Imagen 1 (Solar Day)
    solar_day_seq = [
        "#3A9E5E","#63B66B","#8CCB75","#B2DB7E","#D4E583","#F0E987","#F8D679",
        "#F4B85E","#EC944B","#DE6D3D","#CB4B34"
    ]

    if name == 'solar_day':
        seq = solar_day_seq
    else:  # 'pvout' o cualquier otro -> fallback a pvout
        seq = pvout_seq

    # Re-muestreo a n colores (en sRGB; luego se filtra en RGB lineal)
    m = len(seq)
    if m == n:
        return seq[:]
    out=[]
    def hex_to_rgb(h:str)->Tuple[float,float,float]:
        h=h.lstrip('#')
        return (int(h[0:2],16)/255.0, int(h[2:4],16)/255.0, int(h[4:6],16)/255.0)
    for i in range(n):
        u = i/(n-1)
        pos = u*(m-1)
        j = int(math.floor(pos))
        k = min(j+1, m-1)
        t = pos - j
        r1,g1,b1 = hex_to_rgb(seq[j]); r2,g2,b2 = hex_to_rgb(seq[k])
        r=_interp(r1,r2,t); g=_interp(g1,g2,t); b=_interp(b1,b2,t)
        out.append(_hex(r,g,b))
    return out

# ---- Filtro de accesibilidad (idempotente sobre la paleta base) --------------

def _filter_hex_list(hex_list: List[str], mode: Mode) -> List[str]:
    if mode == 'none':
        return hex_list[:]  # idempotente: no tocar
    out=[]
    for h in hex_list:
        R = int(h[1:3],16)/255.0
        G = int(h[3:5],16)/255.0
        B = int(h[5:7],16)/255.0
        # sRGB -> linear
        rL = _srgb_to_linear(R); gL = _srgb_to_linear(G); bL = _srgb_to_linear(B)
        if mode == 'achroma':
            r2, g2, b2 = _to_gray_lin((rL,gL,bL))
        else:
            M = M_PROTAN if mode=='protan' else (M_DEUTAN if mode=='deutan' else M_TRITAN)
            r2, g2, b2 = _apply_matrix_lin((rL,gL,bL), M)
        # clamp linear
        r2=_clamp01(r2); g2=_clamp01(g2); b2=_clamp01(b2)
        # linear -> sRGB
        rS=_linear_to_srgb(r2); gS=_linear_to_srgb(g2); bS=_linear_to_srgb(b2)
        out.append(_hex(rS,gS,bS))
    return out

def _to_plotly_colorscale(hex_list: List[str]) -> List[Tuple[float, str]]:
    n=len(hex_list)
    if n<=1:
        return [(0.0, hex_list[0] if hex_list else '#000000'), (1.0, hex_list[0] if hex_list else '#000000')]
    return [(i/(n-1), hex_list[i]) for i in range(n)]

# ---- API pedida --------------------------------------------------------------

def setColorBlindMode(enabled: bool, type: Mode) -> Dict[str, Any]:
    """Solo retorna un state dict representando la selección actual."""
    mode: Mode = 'none' if (not enabled or type=='none') else type
    return {'enabled': enabled, 'mode': mode}

def setColorMap(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    options: {
      base: 'pvout'|'solar_day'|list[str],
      enabled: bool,
      mode: Mode
    }
    Devuelve { 'colorscale': plotly_colorscale, 'native': list[str] }
    """
    base = options.get('base','pvout')  # <- PVOUT por defecto
    mode = options.get('mode','none')
    enabled = bool(options.get('enabled', False))

    base_hex = _sample_base_palette(base, n=256)   # sin banding
    filtered_hex = _filter_hex_list(base_hex, mode if enabled else 'none')
    return {
        'colorscale': _to_plotly_colorscale(filtered_hex),
        'native': filtered_hex
    }
