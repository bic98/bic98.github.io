---
# Page settings
layout: default
keywords:
comments: true

# Hero section
title: QGIS tutorial
description: QGIS tutorial for beginners

# # Author box
# author:
#     title: About Author
#     title_url: '#'
#     external_url: true
#     description: Author description

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/reinforce/'
    next:
        content: Next page
        url: '/o3p/'

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

