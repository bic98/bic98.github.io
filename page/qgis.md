---
# Page settings
layout: default
keywords: QGIS, GIS, 지리정보시스템, 지도, 공간정보, 인찬백, InchanBaek, 지오데이터, 지리공간정보, 공간분석, geographic information system, spatial analysis, geospatial data, mapping software, open source GIS, QGIS tutorial, QGIS beginners
comments: true
seo:
  title: QGIS Tutorial for Beginners - Complete Guide to GIS | InchanBaek Note
  description: 초보자를 위한 QGIS 완벽 가이드. 기본 인터페이스부터 공간 분석, 지도 제작까지 단계별 설명으로 지리정보시스템의 핵심 기능을 쉽게 배울 수 있습니다.
  canonical: https://bic98.github.io/qgis/
  image: https://bic98.github.io/images/qgis_avl.png

# Hero section
title: QGIS Tutorial for Beginners
description: 지리정보시스템의 기초부터 고급 분석 기법까지, 무료 오픈소스 GIS 소프트웨어인 QGIS를 활용하는 초보자를 위한 완벽 가이드입니다.

# # Author box
# author:
#     title: About Author
#     title_url: '#'
#     external_url: true
#     description: Author description

# Micro navigation
micro_nav: true

# Page navigation
# page_nav:
#     prev:
#         content: Previous page
#         url: '/reinforce/'
#     next:
#         content: Next page
#         url: '/o3p/'

# Language setting
---

## GIS란 무엇인가?

- GIS는 지리정보시스템(Geographic Information System)의 약자로, 지리적 데이터를 수집, 저장, 관리, 분석, 표현하는 시스템을 말한다.
- 생활에 필요한 다양한 정보를 지도로 표현하여 제공하고, 이를 통해 지리적 정보를 분석하고 활용할 수 있다.

### GIS의 데이터 종류

- **벡터 데이터(Vector Data)**
    - 점(Point), 선(Line), 면(Polygon) 등의 기하학적 요소로 표현되는 데이터
    - 지리적 데이터를 정확하게 표현할 수 있으며, 작은 데이터 용량으로 표현 가능
    - 확대 및 축소에 따라 해상도가 변하지 않는다.
    - 한국에서 제공하는 gis의 대부분은 벡터 데이터로 제공된다.
- **라스터 데이터(Raster Data)**
    - 픽셀(Pixel) 단위로 표현되는 데이터, 주로 위성사진이나 지도 등에 사용
    - 지리적 데이터를 정확하게 표현하기 어렵지만, 다양한 정보를 표현할 수 있다.
    - 확대 및 축소에 따라 해상도가 변한다. 따라서 해상도에 따라 이미지의 선명도가 달라진다.


| 데이터 유형 | 표현 방식 | 특징 | 사용 예시 |
|------------|-----------------|--------------------------------|------------------|
| **벡터 (Vector)** | 점, 선, 면 | - 정확한 위치 표현 <br> - 용량 작음 <br> - 확대해도 선명 | 도로, 행정구역 |
| **라스터 (Raster)** | 픽셀 | - 해상도 영향 받음 <br> - 다양한 정보 포함 | 위성사진, 지형도 |



### GIS의 좌표계
좌표계는 지구상의 위치를 나타내기 위한 좌표계로, 지구는 3차원이지만, 지도는 2차원으로 표현된다. 따라서 지구상의 위치를 2차원 지도에 표현하기 위해 좌표계가 필요하다.
좌표계는 크게 지리 좌표계와 투영 좌표계로 나뉜다.

- **지리 좌표계(Geographic Coordinate System)**
    - 지구의 위경도를 사용하여 위치를 나타내는 좌표계
    - 위도와 경도로 표현되며, 단위는 도, 분, 초로 표현된다.
    - WGS84, GRS80 등이 있다.
    - 위도와 경도는 각각 90도, 180도까지 표현되며, 위도는 적도를 기준으로 북위와 남위로 나뉜다. 경도는 본초자오선을 기준으로 동경과 서경으로 나뉜다.


<div align="center">
    <img src="https://media.licdn.com/dms/image/v2/D5612AQGoitcDV2PdUw/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1664039155968?e=1746662400&v=beta&t=GpsJb76-okrYenY61dDqib8ZdAO3rMY_036jyip_P24" alt="matrix multiplication" width="300">
</div>
<br>
- **투영 좌표계(Projection Coordinate System)**
    - 지구의 3차원 표면을 2차원 지도에 투영하는 좌표계
    - 직각좌표계, 원통좌표계, 원뿔좌표계 등이 있다.
    - 한국에서는 주로 UTM좌표계를 사용한다. UTM좌표계는 원뿔좌표계로 경도 6도 단위로 나누어 총 60개의 UTM존으로 구분한다. 한국을 6개의 존으로 나누어 표현한다. UTM좌표계는 1존에 대한 좌표계이므로, 한국은 52~53존에 걸쳐있으므로, 52존에 대한 좌표계와 53존에 대한 좌표계로 나뉜다.  


<br>
<div align="center">
    <img src="https://www.usu.edu/geospatial/images/tutorials/core-concepts/projected-coordinate-systems/cc5-projection-surface.png" alt="matrix multiplication" width="400">
</div>

| 좌표계 유형 | 설명 | 특징 | 사용 예시 |
|------------|-------------------------|--------------------------------|------------------|
| **지리 좌표계 (GCS)** | 위도·경도로 위치 표현 | - 단위: 도(°), 분('), 초(") <br> - 3차원 좌표 <br> - WGS84, GRS80 사용 | GPS, 위경도 기반 데이터 |
| **투영 좌표계 (PCS)** | 3D 지구를 2D로 투영 | - 직각/원통/원뿔 투영 <br> - 단위: 미터(m) <br> - UTM 좌표계 사용 | 지도 제작, GIS 분석 |

### 언제 어떤 좌표계를 사용해야 하는가?

- **EPSG코드: 좌표계, 투영법, 변환법을 표준화**
    - 좌표계를 표현하기 위한 코드로, 좌표계의 정보를 담고 있다.
    - 좌표계를 사용할 때, EPSG코드를 사용하여 좌표계를 지정할 수 있다.
    - 한국에서는 주로 EPSG:5179를 사용한다. EPSG:5179는 UTM-K좌표계로, 한국을 표현하기 위한 좌표계이다.

| EPSG 코드 | 좌표계 이름 | 설명 |
|-----------|-----------------|--------------------------------|
| **4326**  | WGS 84 | GPS에서 사용하는 지리 좌표계 (경도, 위도) |
| **3857**  | Pseudo-Mercator | Google Maps에서 사용하는 웹 좌표계 |
| **5179**  | Korea 2000 | 한국 TM(Transverse Mercator) 좌표계 |
| **32652** | UTM Zone 52N | UTM 52N 투영 좌표계 (미터 단위) |
| **4019**  | GRS80 | 국제 측량 표준 타원체 |

### GIS 데이터 제공사이트

- **공공데이터포털**
    - [공공데이터포털](https://www.data.go.kr/)
    - 한국에서 제공하는 공공데이터를 제공하는 사이트
    - 행정구역, 지형도, 교통망, 위성사진 등 다양한 정보를 제공한다.
- **서울 열린데이터광장**
    - [열린데이터광장](https://data.seoul.go.kr/)
    - 서울시에서 제공하는 열린데이터를 제공하는 사이트
    - 서울시의 다양한 정보를 제공하며, API를 통해 데이터를 사용할 수 있다.
- **브이월드**
    - [브이월드](http://www.vworld.kr/)
    - 한국지도를 제공하는 사이트
    - 다양한 정보를 제공하며, API를 통해 지도를 사용할 수 있다.
- **국토지리정보원**
    - [국토지리정보원](https://map.ngii.go.kr/ms/map/NlipMap.do)
    - 국토지리정보원에서 제공하는 지리정보를 제공하는 사이트
    - 행정구역, 지형도, 교통망, 위성사진 등 다양한 정보를 제공한다.
- **AI Hub**
    - [AI Hub](http://www.aihub.or.kr/)
    - 한국에서 제공하는 인공지능 데이터를 제공하는 사이트
    - 음성, 이미지, 텍스트 등 다양한 데이터를 제공한다.
- **KOSIS국가통계포털**
    - [KOSIS](http://kosis.kr/)
    - 국가통계를 제공하는 사이트
    - 국가통계를 제공하며, 다양한 통계를 제공한다.
- **행정안전부 행정정보시스템**
    - [행정정보시스템](https://jumin.mois.go.kr/#)
    - 행정정보를 제공하는 사이트
    - 행정정보를 제공하며, 다양한 정보를 제공한다.
    - 주민등록, 주소, 건물 등 다양한 정보를 제공한다.

## python에서 GIS 데이터 다루기

qgis에서 사용하는 데이터는 shp파일이다. shp파일은 벡터 데이터로, 지리정보를 다루는데 사용된다. python에서 shp파일을 다루기 위해서는 geopandas를 사용한다.

QgsVectorLayer() 함수를 사용하여 shp파일을 불러올 수 있다.

```python   
import qgis.core

# shp파일을 불러온다.
layer = QgsVectorLayer('data/TL_SCCO_SIG.shp', 'layer_name', 'ogr')

# 레이어를 추가한다.
QgsProject.instance().addMapLayer(layer)
``` 

- **QgsVectorLayer()**
    - shp파일을 불러오는 함수
    - shp파일의 경로, 레이어 이름, 데이터 포맷을 입력하여 shp파일을 불러온다.
    - shp파일의 경로는 절대경로나 상대경로로 입력할 수 있다.
    - 레이어 이름은 레이어의 이름을 지정하는 것으로, 레이어 이름을 지정하지 않으면 shp파일의 이름으로 레이어 이름이 지정된다.
    - 데이터 포맷은 shp파일의 데이터 포맷을 지정하는 것으로, 'ogr'을 입력하면 shp파일을 불러올 수 있다.

- **QgsProject.instance().addMapLayer()**
    - 레이어를 추가하는 함수
    - 레이어를 추가하면 qgis에 레이어가 추가된다.
    - 레이어를 추가하면 qgis에서 레이어를 확인할 수 있다.

addVectorLayer() 함수를 사용하면 레이어를 생성하고 바로 맵에 추가할 수 있다.

```python
# shp파일을 불러온다.
layer = iface.addVectorLayer('data/TL_SCCO_SIG.shp', 'layer_name', 'ogr')
``` 

- **iface.addVectorLayer()**
    - shp파일을 불러오는 함수
    - shp파일의 경로, 레이어 이름, 데이터 포맷을 입력하여 shp파일을 불러온다.
    - shp파일의 경로는 절대경로나 상대경로로 입력할 수 있다.
    - 레이어 이름은 레이어의 이름을 지정하는 것으로, 레이어 이름을 지정하지 않으면 shp파일의 이름으로 레이어 이름이 지정된다.
    - 데이터 포맷은 shp파일의 데이터 포맷을 지정하는 것으로, 'ogr'을 입력하면 shp파일을 불러올 수 있다.

**ogr**은 **OGR Simple Feature Library**의 약자로, 벡터 데이터를 다루는 라이브러리이다. **ogr**은 다양한 데이터 포맷을 지원하며, 벡터 데이터를 다루는데 사용된다.

파일을 불러오면 레이어의 정보를 확인할 수 있다. 하지만 코드를 더할때마다 계속해서 레이어를 불러오게 되면, 레이어가 중복되어 레이어가 계속해서 추가된다. 따라서 레이어를 추가하기 전에 레이어가 이미 있는지 확인해야 한다.

```python
# 경로 설정
shp_path = r"C:\Users\@@@@@\Downloads\(B100)국토통계_인구정보-총 인구 수(전체)-읍면동경계_서울특별시_202410\nlsp_003001001.shp"
layer_name = '행정동별인구'

# 이미 해당 이름의 레이어가 있는지 확인
layer_exists = False
layers = QgsProject.instance().mapLayersByName(layer_name)
if layers:
    layer_exists = True

# 레이어가 없는 경우에만 추가
if not layer_exists:
    iface.addVectorLayer(shp_path, layer_name, 'ogr')
else:
    print(f"레이어 '{layer_name}'이(가) 이미 존재합니다.")
```

<div align="center">
    <img src="/images/qgis_avl.png" alt="qgis" width = "500">
</div>

---

### 필드를 가지고 오는 방법

```python
# 레이어를 불러온다.
layer = iface.addVectorLayer('data/TL_SCCO_SIG.shp', 'layer_name', 'ogr')

# 필드를 가지고 온다.
fields = layer.fields()

# 필드를 출력한다.
for field in fields:
    print(field.name())
```

근데 위와 같이 addVectorLayer()를 사용하면 실행할때마다 레이어가 중복되어 추가된다. 그래서 QgsVectorLayer()를 사용하여 레이어를 불러온다.

```python
# shp파일을 불러온다.
layer = QgsVectorLayer('data/TL_SCCO_SIG.shp', 'layer_name', 'ogr')

fields = layer.fields()

# 필드를 출력한다.
for field in fields:
    print(field.name())
``` 

### 필드와 데이터를 판다스 데이터프레임으로 변환

데이터 정제와 분석을 위해 필드와 데이터를 판다스 데이터프레임으로 변환할 수 있다.

```python
import pandas as pd
df = pd.DataFrame([feat.attributes() for feat in vlayer.getFeatures()],
                  columns=[field.name() for field in vlayer.fields()])
print(df)
```

위의 코드에서 feat.attributes()는 레이어의 속성을 리스트 형태로 반환한다.
field.name()은 레이어의 필드 이름을 반환한다. 필드 이름을 데이터 프레임의 열 이름으로 사용한다.

```python
print(df)
gid       lbl      val
0    11290110   2435.00   2435.0
1    11140101      8.00      8.0
2    11110152     54.00     54.0
3    11410111  55081.00  55081.0
4    11650109   7877.00   7877.0
..                   ...
462  11560131      NULL     NULL
463  11170135      NULL     NULL
464  11500111      NULL     NULL
465  11110127      NULL     NULL
466  11140106      NULL     NULL
```
---

### 데이터프레임에서 속성 타입 확인 및 변환

데이터프레임(df)에서 속성들이 문자열인지 실수형인지 확인하고 변환하는 방법은 다음과 같다.

1. 속성 타입 확인:
```python
print(df.dtypes)
```

2. 문자열을 실수형으로 변환:
```python
df['속성명'] = df['속성명'].astype(float)
```

예시:
```python
import pandas as pd

# 예제 데이터프레임 생성

```python
data = {
    'gid': [0, 1, 2, 3, 4],
    'lbl': ['11290110', '11140101', '11110152', '11410111', '11650109'],
    'val': ['2435.00', '8.00', '54.00', '55081.00', '7877.00']
}

df = pd.DataFrame(data)

# 속성 타입 확인
print(df.dtypes)

# 문자열을 실수형으로 변환
df['val'] = df['val'].astype(float)

# 변환 후 속성 타입 확인
print(df.dtypes)

print(df)
```

만약에 null값이 포함되어 있으면, null값을 0으로 변환하거나 null값을 제거해야 한다.

```python
# null값을 0으로 변환
df['val'] = df['val'].fillna(0)

# null값을 제거
df = df.dropna()
``` 

하지만 그래도 오류가 나는 경우에는 다음과 같은 방법을 사용한다. 

```python
import pandas as pd
import numpy as np

# 먼저 null 값 확인
print("변환 전 null 값 개수:")
print(df.isnull().sum())

# 문자열을 숫자로 변환 시도 (변환 불가능한 값은 NaN으로 처리)
df['gid'] = pd.to_numeric(df['gid'], errors='coerce')
df['lbl'] = pd.to_numeric(df['lbl'], errors='coerce')

# 변환 후 null 값 확인
print("\n변환 후 null 값 개수:")
print(df.isnull().sum())

# 방법 1: null 값을 0으로 치환
df['gid'] = df['gid'].fillna(0)
df['lbl'] = df['lbl'].fillna(0)

# 또는
# 방법 2: null 값을 평균값으로 치환
# df['gid'] = df['gid'].fillna(df['gid'].mean())
# df['lbl'] = df['lbl'].fillna(df['lbl'].mean())

# 최종 확인
print("\n치환 후 데이터 타입:")
print(df.dtypes)
print("\n남은 null 값 개수:")
print(df.isnull().sum())
```

따라서 지금까지 모든 코드를 합치면 다음과 같다.

```python
# 경로 설정
shp_path = r"C:\Users\BaekInchan\Downloads\(B100)국토통계_인구정보-총 인구 수(전체)-읍면동경계_서울특별시_202410\nlsp_003001001.shp"
layer_name = '행정동별인구'

# 이미 해당 이름의 레이어가 있는지 확인
layer_exists = False
layers = QgsProject.instance().mapLayersByName(layer_name)
if layers:
    layer_exists = True

# 레이어가 없는 경우에만 추가
if not layer_exists:
    iface.addVectorLayer(shp_path, layer_name, 'ogr')
else:
    print(f"레이어 '{layer_name}'이(가) 이미 존재합니다.")

#레이어 필드 확인
vlayer = QgsVectorLayer(shp_path, layer_name, 'ogr')
fields = vlayer.fields()
for field in fields:
    field_name = field.name()
    field_type = field.typeName()
    print(field_name, field_type)
    
df = pd.DataFrame([feat.attributes() for feat in vlayer.getFeatures()],
                  columns=[field.name() for field in vlayer.fields()])

#print([feat.attributes() for feat in vlayer.getFeatures()])
df['gid'] = pd.to_numeric(df['gid'], errors='coerce')
df['lbl'] = pd.to_numeric(df['lbl'], errors='coerce')
df['val'] = pd.to_numeric(df['val'], errors='coerce')
print(df)
print(df.dtypes)
df['val'] = df['val'].fillna(0)
df['lbl'] = df['lbl'].fillna(0)
print(df.isna().sum())
print(df)

```
## qgis push, pull 방법
shp파일을 불러오고, 엑셀이나 csv파일로 만들수 있지 않을까? 그 반대로 엑셀이나 csv파일을 불러와서 shp파일로 만들수 있지 않을까? 이때, qgis에서는 push, pull 방법을 사용한다.

정식적인 명칭은 아니고, git에서 사용하는 push, pull 방법을 사용하여 데이터를 불러오고, 저장하는 방법이다. 이 방법을 비슷하게 가지고와서 만들어보았다. 

코드는 다음과 같다. 

```python
import pandas as pd
import os
from functools import singledispatch
from typing import Optional, Union, List
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsField, QgsFeature, 
    QgsGeometry, QgsCoordinateReferenceSystem, QgsVectorFileWriter
)
from qgis.PyQt.QtCore import QVariant



# 모든 사용자에게 적용 가능한 경로 설정
base_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "QGIS")

# 폴더가 없으면 자동 생성
os.makedirs(base_path, exist_ok=True)
# Default coordinate reference system
GLOBAL_CRS = "EPSG:5179"  # Korean coordinate system

@singledispatch
def push(data, layer_name: str) -> None:
    """Base function for pushing data to QGIS layers."""
    raise NotImplementedError("지원하지 않는 데이터 타입입니다.")


@push.register(str)
def _(shp_path: str, layer_name: str) -> Optional[QgsVectorLayer]:
    """
    Register a shapefile as a QGIS layer.
    
    Args:
        shp_path: Path to the shapefile
        layer_name: Name for the QGIS layer
        
    Returns:
        QgsVectorLayer or None: The created/existing layer or None on failure
    """
    # Check if layer already exists
    existing = QgsProject.instance().mapLayersByName(layer_name)
    if existing:
        print(f"레이어 '{layer_name}' 이미 존재 (재등록하지 않음)")
        return existing[0]
        
    # Create new layer from shapefile
    layer = QgsVectorLayer(shp_path, layer_name, 'ogr')
    if not layer.isValid():
        print("레이어 등록 실패: 경로 확인")
        return None
        
    # Set coordinate system and add to project
    layer.setCrs(QgsCoordinateReferenceSystem(GLOBAL_CRS))
    QgsProject.instance().addMapLayer(layer)
    print(f"Shapefile 레이어 '{layer_name}' 등록 완료")
    return layer


@push.register(pd.DataFrame)
def _(df: pd.DataFrame, layer_name: str) -> Optional[QgsVectorLayer]:
    """
    Register a pandas DataFrame as a QGIS layer with geometry support.
    If geometry column (WKT) exists, saves as shapefile, otherwise as CSV.
    
    Args:
        df: DataFrame containing data and optional WKT geometry column
        layer_name: Name for the QGIS layer
        
    Returns:
        QgsVectorLayer or None: The created layer or None on failure
    """
    # Check if layer already exists
    existing = QgsProject.instance().mapLayersByName(layer_name)
    if existing:
        print(f"레이어 '{layer_name}' 이미 존재 (재등록하지 않음)")
        return existing[0]
    
    # Detect geometry column and type
    geom_col = 'WKT' if 'WKT' in df.columns else None
    geometry_type = "None"
    
    if geom_col:
        # Get first valid geometry to determine type
        valid_geoms = df.loc[df[geom_col].notnull(), geom_col]
        if not valid_geoms.empty:
            sample_geom = QgsGeometry.fromWkt(valid_geoms.iloc[0])
            geometry_type = {
                0: "Point",
                1: "LineString",               2: "Polygon"
            }.get(sample_geom.type(), "Unknown")
    
    # 공간 데이터가 있는 경우 Shapefile로 저장
    if geom_col and geometry_type != "None" and geometry_type != "Unknown":
        # Create memory layer with appropriate geometry type
        uri = f"{geometry_type}?crs={GLOBAL_CRS}"
        mem_layer = QgsVectorLayer(uri, layer_name, "memory")
        mem_layer.setCrs(QgsCoordinateReferenceSystem(GLOBAL_CRS))
        provider = mem_layer.dataProvider()

        # Define fields (excluding geometry column)
        fields = []
        for col in df.columns:
            if col != geom_col:
                field_type = QVariant.Double if pd.api.types.is_numeric_dtype(df[col]) else QVariant.String
                fields.append(QgsField(col, field_type))

        provider.addAttributes(fields)
        mem_layer.updateFields()

        # Prepare features in batch (more efficient than one by one)
        features = []
        df_fields = [col for col in df.columns if col != geom_col]

        # Pre-compute indices for performance
        has_geom_mask = df[geom_col].notnull()

        for i, row in enumerate(df[df_fields].values):
            feat = QgsFeature()
            feat.setFields(mem_layer.fields())
            feat.setAttributes(row.tolist())

            # Set geometry if available and valid
            if has_geom_mask.iloc[i]:
                wkt = df[geom_col].iloc[i]
                feat.setGeometry(QgsGeometry.fromWkt(wkt))

            features.append(feat)

        # Add features in batch
        provider.addFeatures(features)
        mem_layer.updateExtents()

        save_path = os.path.join(base_path, f"{layer_name}.shp")
        error = QgsVectorFileWriter.writeAsVectorFormat(
            mem_layer, save_path, "UTF-8", mem_layer.crs(), "ESRI Shapefile"
        )
        if error[0] != QgsVectorFileWriter.NoError:
            print("파일 저장 중 오류 발생:", error)
            return None
        print(f"레이어가 '{save_path}'에 저장되었습니다.")

        # 저장된 파일을 다시 불러오기
        saved_layer = QgsVectorLayer(save_path, layer_name, "ogr")
        QgsProject.instance().addMapLayer(saved_layer)
        print(f"레이어 '{layer_name}'이(가) 공간 데이터로 등록됨. 피처 수: {saved_layer.featureCount()}")
        return saved_layer

    else:
        # 공간 데이터가 없는 경우 CSV로 저장
        csv_path = os.path.join(base_path, f"{layer_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"데이터가 '{csv_path}'에 CSV로 저장되었습니다.")

        # 경로 슬래시 방향 통일 (Windows 백슬래시 문제 해결)
        csv_path_uri = csv_path.replace('\\', '/')

        # CSV 파일을 레이어로 불러오기 - URI 형식 수정
        uri = f"file:///{csv_path_uri}?delimiter=,&encoding=UTF-8&type=csv&detectTypes=yes"

        # 디버깅 정보 출력
        print(f"CSV 레이어 URI: {uri}")

        csv_layer = QgsVectorLayer(uri, layer_name, "delimitedtext")

        if not csv_layer.isValid():
            print("CSV 레이어를 생성할 수 없습니다. 오류 메시지:")
            try:
                print(csv_layer.error().summary())
            except:
                print("상세 오류 정보를 가져올 수 없습니다.")
                
            # 대안: 테이블 레이어로 직접 생성 시도
            print("대안 방법으로 데이터프레임을 메모리 테이블로 직접 등록합니다.")
            uri = "none"
            table_layer = QgsVectorLayer(uri, layer_name, "memory")
            provider = table_layer.dataProvider()

            # 필드 정의
            fields = []
            for col in df.columns:
                field_type = QVariant.Double if pd.api.types.is_numeric_dtype(df[col]) else QVariant.String
                fields.append(QgsField(col, field_type))

            provider.addAttributes(fields)
            table_layer.updateFields()

            # 데이터 추가
            features = []
            for idx, row in df.iterrows():
                feat = QgsFeature()
                feat.setFields(table_layer.fields())
                feat.setAttributes(row.tolist())
                features.append(feat)

            provider.addFeatures(features)
            QgsProject.instance().addMapLayer(table_layer)
            print(f"레이어 '{layer_name}'이(가) 메모리 테이블로 등록됨. 행 수: {table_layer.featureCount()}")
            return table_layer
        else:
            QgsProject.instance().addMapLayer(csv_layer)
            print(f"레이어 '{layer_name}'이(가) CSV 테이블 데이터로 등록됨. 행 수: {csv_layer.featureCount()}")
            return csv_layer

def pull(layer_name):
    existing = QgsProject.instance().mapLayersByName(layer_name)
    if not existing:
        print(f"레이어 '{layer_name}'가 존재하지 않습니다.")
        return None
    layer = existing[0]
    fields = layer.fields()
    print("필드 정보:", " | ".join([f"{f.name()} ({f.typeName()})" for f in fields]))
    
    data = []
    for feat in layer.getFeatures():
        attrs = feat.attributes()
        wkt = feat.geometry().asWkt() if feat.geometry() else None
        attrs.append(wkt)
        data.append(attrs)
    
    col_names = [f.name() for f in fields] + ['WKT']
    df = pd.DataFrame(data, columns=col_names)
    return df

def change_color(layer_name, color):
    """
    레이어 이름을 받아 QGIS 프로젝트에서 해당 레이어를 찾아 심볼 색상을 변경합니다.
    레이어 패널의 심볼 캐시도 함께 업데이트합니다.
    
    Args:
        layer_name (str): 변경할 레이어의 이름.
        color (str or QColor): 새 색상. 예: "#FF0000" 또는 QColor 객체.
    """
    from qgis.PyQt.QtGui import QColor
    from qgis.core import QgsProject
    from qgis.utils import iface
    
    # color가 문자열이면 QColor로 변환
    if isinstance(color, str):
        color = QColor(color)
    
    # QgsProject에서 레이어 검색
    layers = QgsProject.instance().mapLayersByName(layer_name)
    if not layers:
        print(f"레이어 '{layer_name}'을(를) 찾을 수 없습니다.")
        return
    layer = layers[0]
    
    renderer = layer.renderer()
    if renderer is None:
        print("레이어에 렌더러가 없습니다.")
        return
    
    symbol = renderer.symbol()
    symbol.setColor(color)
    
    # 레이어 다시 그리기
    layer.triggerRepaint()
    
    # 레이어 패널의 심볼 캐시 업데이트
    iface.layerTreeView().refreshLayerSymbology(layer.id())
    
    # 레이어 범례 업데이트
    if hasattr(iface, 'legendInterface'):
        iface.legendInterface().refreshLayerSymbology(layer)
    
    print(f"레이어 '{layer.name()}'의 색상이 {color.name()}(으)로 변경되었습니다.")

def remove(layer_name: str) -> None:
    """
    QGIS 프로젝트에서 지정한 이름의 레이어를 모두 제거합니다.
    
    Args:
        layer_name (str): 제거할 레이어의 이름.
    """
    layers = QgsProject.instance().mapLayersByName(layer_name)
    if not layers:
        print(f"레이어 '{layer_name}'가 존재하지 않습니다.")
        return

    for layer in layers:
        QgsProject.instance().removeMapLayer(layer.id())
    print(f"레이어 '{layer_name}'가 제거되었습니다.")

```


사용하는 방법은 다음과 같다.

C:\Users\XXXXXXXXXX\
AppData\Roaming\QGIS\
QGIS3\profiles\default\
python\plugins

위의 경로에 코드를 저장하고, qgis를 실행하면, push, pull, remove, change_color 함수를 사용할 수 있다.

필자는 본인의 이름을 따서 inchan.py로 저장하였다. 

그럼 qgis 파이썬 플로그인에서 

```python
import inchan
#inchan.GLOBAL_CRS = "EPSG:5179"(기본값) -> 한국 좌표계(필요에 따라 변경 가능)
#inchan.push("data/TL_SCCO_SIG.shp", "layer_name")
#inchan.push(df, "layer_name")
#inchan.pull("layer_name")
#inchan.remove("layer_name")
#inchan.change_color("layer_name", "#FF0000")
```
push 함수는 shp파일을 불러오거나, 데이터프레임을 레이어로 만들어준다.
shp파일과 엑셀, csv파일의 차이점은 wkt라는 컬럼이 있는지 없는지이다. wkt라는 컬럼이 있으면 shp파일로 저장하고, 없으면 csv파일로 저장한다.

pull 함수는 레이어의 정보를 데이터프레임으로 변환한다.
판다스 데이터프레임으로 변환하면, 데이터를 분석하거나, 시각화할 수 있기 때문이다. 즉, shp파일의 정보를 판다스 데이터 프레임으로 변환하여 데이터를 정제, 분석, 조인등을 쉽게 하여 다시 푸쉬하면 일을 정말 쉽게 할 수 있다.

remove 함수는 레이어를 제거한다.

change_color 함수는 레이어의 색상을 변경한다.

이렇게 사용하면 된다.



## 행정동 코드를 행정동 이름 변환

gid는 행정동 코드로 되어있다. 우리가 사용하기에는 코드번호보다는 문자열로된 행정동 이름이 더 편리하다. 따라서 gid를 행정동 이름으로 변환해야 한다. 이때, 행정동 코드와 행정동 이름이 매칭된 데이터를 사용하면 된다.
