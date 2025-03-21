---
# Page settings
layout: default
keywords: 풀스택, 웹개발, 프론트엔드, 백엔드, 데이터베이스, 인찬백, InchanBaek, 풀스택 개발자, 자바스크립트, 파이썬, React, Node.js, Django, Full Stack, Web Development, Frontend, Backend, Database
comments: true
seo:
  title: 풀스택 개발 정리 | InchanBaek Note
  description: 프론트엔드와 백엔드를 아우르는 풀스택 개발 기술과 지식을 정리한 노트입니다. 웹 개발의 모든 영역을 다루는 실용적인 내용을 담고 있습니다.
  canonical: https://bic98.github.io/fullstack/
  image: https://bic98.github.io/images/layout/logo.png

# Hero section
title: 풀스택 개발 정리
description: 프론트엔드부터 백엔드까지, 풀스택 개발에 필요한 기술과 지식을 정리한 노트입니다.

# # Author box
# author:
#     title: About Author
#     title_url: '#'
#     external_url: true
#     description: Author description

# Micro navigation
micro_nav: true

# Page navigation

# Language setting
---

## 풀스택 개발이란?

풀스택 개발은 웹 애플리케이션의 프론트엔드(사용자 인터페이스)와 백엔드(서버 로직, 데이터베이스) 모두를 개발할 수 있는 능력을 의미한다. 

### 프론트엔드 기술

- **HTML/CSS/JavaScript**: 웹의 기본 구성 요소
- **프레임워크/라이브러리**: React, Vue.js, Angular
- **상태 관리**: Redux, Vuex, Context API
- **CSS 프레임워크**: Bootstrap, Tailwind CSS
- **타입스크립트**: 정적 타입을 지원하는 자바스크립트 확장

### 백엔드 기술

- **서버 언어**: Node.js, Python, Java, Go
- **프레임워크**: Express, Django, Spring Boot, Flask
- **데이터베이스**: MySQL, PostgreSQL, MongoDB, Redis
- **API 설계**: REST, GraphQL
- **인증/보안**: JWT, OAuth, HTTPS

### 개발 도구 및 환경

- **버전 관리**: Git, GitHub, GitLab
- **배포**: Docker, Kubernetes, AWS, Heroku
- **CI/CD**: Jenkins, GitHub Actions, Travis CI
- **테스트**: Jest, Mocha, Pytest
- **개발 환경**: VS Code, IntelliJ, Postman


## 서버와 클라이언트

요청이라는 행위의 주체를 기준으로 클라이언트와 서버로 나눌 수 있다. 클라이언트는 사용자가 요청을 보내는 주체이며, 서버는 요청을 받아 처리하는 주체이다. 

웹개발에서 클라이언트는 웹 브라우저를 의미하며 구글 크롬, 파이어폭스 사파이가 있다.  서버는 웹 서버를 의미한다. 대표적인 웹서버는 NGINX, Apache등이 있다. 클라이언트는 사용자가 보는 화면을 담당하고, 서버는 데이터를 처리하고 저장하는 역할을 한다. 

### 서버의 주소 DNS, IP

서버는 인터넷에 연결되어 있어야 클라이언트가 요청을 보낼 수 있다. 서버는 고유한 주소를 가지고 있으며, 이 주소를 통해 클라이언트가 서버에 접속할 수 있다.

도메인 네임 시스템(DNS)은 도메인 이름을 IP 주소로 변환하는 시스템이다. 사용자가 도메인 이름을 입력하면 DNS 서버가 해당 도메인 이름에 대응하는 IP 주소를 찾아 클라이언트에게 전달한다. 클라이언트는 IP 주소를 통해 서버에 접속한다.

### 포트

웹 브라우저는 DNS를 통해 서버의 IP 주소를 알아내면, 해당 서버에 접속한다. 서버는 여러 개의 서비스를 제공할 수 있으며, 각 서비스는 포트 번호로 구분된다.

포트 번호는 0부터 65535까지 사용할 수 있으며, 0~1023번까지는 잘 알려진 포트로 사용되고 있다. HTTP 서버는 80번 포트를 사용하며, HTTPS 서버는 443번 포트를 사용한다.

대표적인 프로토콜과 포트 번호는 다음과 같다.
(프로토콜이란? 컴퓨터나 네트워크 장비 사이에서 메시지를 주고받는 양식이다.)

- HTTP: 80 -> 웹에서 HTML등 다양한 데이터를 주고받을 수 있는 프로토콜
- HTTPS: 443 -> HTTP의 보안 버전
- FTP: 21 -> 파일 전송 프로토콜
- SSH: 22 -> 원격 접속 프로토콜
- SMTP: 25  -> 이메일 전송 프로토콜
- POP3: 110 -> 이메일 수신 프로토콜

### HTTP 프로토콜

HTTP(HyperText Transfer Protocol)는 클라이언트와 서버 간에 데이터를 주고받기 위한 프로토콜이다. HTTP는 요청-응답 방식으로 동작하며, 클라이언트가 요청을 보내면 서버가 응답을 보낸다.
자주 사용되는 HTTP 요청 메소드는 다음과 같다.

- GET: 서버에서 데이터를 가져올 때 사용하는 메소드
- POST: 서버에 데이터를 전송할 때 사용하는 메소드
- PUT: 서버에 데이터를 업데이트할 때 사용하는 메소드
- DELETE: 서버에 데이터를 삭제할 때 사용하는 메소드

HTTP 응답 코드는 서버가 클라이언트에게 응답할 때 상태를 알려주는 코드이다. 대표적인 응답 코드는 다음과 같다.

- 200: OK -> 요청이 성공했을 때
- 201: Created -> 요청이 성공하고 새로운 리소스가 생성되었을 때
- 400: Bad Request -> 요청이 잘못되었을 때
- 401: Unauthorized -> 인증이 필요한 요청일 때
- 404: Not Found -> 요청한 리소스가 없을 때
- 500: Internal Server Error -> 서버 내부 오류가 발생했을 때

### http 쿠키

HTTP 쿠키는 서버가 클라이언트에게 전달하는 작은 데이터 조각이다. 쿠키는 클라이언트의 로컬에 저장되며, 서버는 클라이언트에게 쿠키를 전달하여 클라이언트의 상태를 유지할 수 있다.
예를들어, 웹사이트에 접속하였을 때, '오늘 하루 동안 보지 않기'라는 버튼을 누르면 쿠키가 클라이언트에 저장되어, 다음에 접속했을 때 더 이상 팝업창이 뜨지 않는다. 즉, 다시 사용자가 동일한 사이트에 접속했을 때, 팝업 설정 데이터가 쿠키에 있다면 팝업창이 뜨지 않는다.

쿠키는 다음과 같은 특징이 있다.

- 이름 = 값 형태로 저장된다.
- 도메인, 경로, 만료 날짜 등의 속성을 가질 수 있다.
- 클라이언트의 로컬에 저장되어 있다.
- 서버와 클라이언트 간의 상태를 유지할 수 있다.


## 프론트엔드 개발도구

프론트엔드 개발을 위한 다양한 도구와 환경을 소개한다.

### 자바스크립트 런타임

- **Node.js**: 자바스크립트 런타임 환경으로, 서버 사이드 개발에 사용된다.

과거에는 자바스크립트가 브라우저에서만 동작했지만, Node.js의 등장으로 자바스크립트 코드를 로컬이나 서버에서 실행할 수 있게 되었다. Node.js는 V8 엔진을 기반으로 만들어졌으며, 비동기 I/O 처리와 이벤트 주도 방식을 지원한다.
(비동기 I/O 처리란? I/O 작업을 기다리지 않고 다른 작업을 수행하는 것을 의미한다., V8 엔진은 구글에서 개발한 자바스크립트 엔진이다.)

### 패키지 매니저

모듈은 프로그램을 구성하는 작은 단위의 요소이자 관련있는 기능들을 묶어놓은 것이다. 패키지는 모듈을 묶어놓은 것이다. 노드 모듈이란 Node.js에서 사용할 수 있는 모듈을 의미한다. 노드 모듈의 종류로는 내장 모듈, 써드 파티 모듈, 사용자 정의 모듈이 있다.

- **내장 모듈**: 노드.js에 기본적으로 포함된 모듈이다.
- **써드 파티 모듈**: 외부에서 만들어진 모듈이다.
- **사용자 정의 모듈**: 사용자가 직접 만든 모듈이다.

패키지 매니저란? 프로그램을 설치하고 관리하는 도구이다. 프론트엔드 개발에서는 주로 npm과 yarn을 사용한다. 

- **npm**: Node.js 패키지 매니저로, 자바스크립트 라이브러리를 설치하고 관리할 수 있다.
- **yarn**: 페이스북에서 만든 패키지 매니저로, npm보다 빠르고 안정적이다.

자주 사용하는 npm 명령어는 다음과 같다.

- `npm init`: 새로운 프로젝트를 시작할 때 사용한다.
- `npm install`: 패키지를 설치할 때 사용한다.
- `npm start`: 프로젝트를 실행할 때 사용한다.
- `npm run build`: 프로젝트를 빌드할 때 사용한다.

자주 사용하는 yarn 명령어는 다음과 같다.

- `yarn init`: 새로운 프로젝트를 시작할 때 사용한다.
- `yarn add`: 패키지를 설치할 때 사용한다.
- `yarn start`: 프로젝트를 실행할 때 사용한다.
- `yarn build`: 프로젝트를 빌드할 때 사용한다.

### 번들러

번들러는 프로젝트에 사용된 모듈을 하나로 묶어주는 도구이다. 번들러를 사용하면 여러 개의 파일을 하나로 합쳐서 네트워크 요청을 줄일 수 있다. 이를 통해 로딩 속도를 개선하고, 코드의 의존성을 관리할 수 있다.

- **Webpack**: 모듈 번들러로, 자바스크립트 파일을 하나로 합쳐준다.
- **Parcel**: 웹 애플리케이션 번들러로, 설정 없이 사용할 수 있다.

### Axios

Axios는 HTTP 클라이언트 라이브러리로, 비동기 방식으로 서버와 데이터를 주고받을 수 있다. Axios는 Promise 기반으로 동작하며, 브라우저와 Node.js에서 모두 사용할 수 있다.

자바스크립트 런타임에서 기본으로 제공하는 Fetch API 와 차이는 다음 표를 보면 알 수 있다. 

| 항목 | Fetch API | Axios |
|---|---|---|
| 브라우저 지원 | IE 11 이상 | 모든 브라우저 지원 |
| 요청 취소 | 불가능 | 가능 |
| 요청/응답 변환 | 번거로움 | 자동으로 처리 |
| 오류 처리 | HTTP 상태 코드만 제공 | HTTP 상태 코드 외에도 오류 메시지 제공 |

(fecth api란? 네트워크 요청을 보내고 응답을 받을 수 있는 API이다. fetch api의 종류로는 Request, Response, Headers 등이 있다.)

### 부트스트랩

부트스트랩은 HTML, CSS, JavaScript로 구성된 오픈 소스 프론트엔드 프레임워크이다. 부트스트랩을 사용하면 웹 페이지를 쉽고 빠르게 디자인할 수 있다. 부트스트랩은 반응형 웹 디자인을 지원하며, 모바일 환경에서도 잘 동작한다.

### Sass

Sass는 CSS 전처리기로, CSS의 기능을 확장하여 사용할 수 있게 해준다. Sass는 변수, 중첩, 믹스인, 상속 등의 기능을 제공하며, 코드의 재사용성을 높일 수 있다.
왜 css 전처리기를 사용하는가? css 전처리기는 css의 한계를 극복하기 위해 사용한다. css는 변수, 중첩, 믹스인, 상속 등의 기능을 제공하지 않기 때문에, css 전처리기를 사용하여 이러한 기능을 사용할 수 있다.

### Scss

Scss는 Sass의 최신 버전으로, CSS와 호환성이 높아 사용하기 쉽다. Scss는 Sass의 모든 기능을 지원하며, CSS와 거의 동일한 문법을 사용한다.

### Prettier

Prettier는 코드 포맷터로, 코드의 일관성을 유지하고 가독성을 높일 수 있다. Prettier는 코드를 자동으로 정렬하고 들여쓰기를 맞춰준다. Prettier는 다양한 언어를 지원하며, VS Code, IntelliJ IDEA 등의 에디터에서 사용할 수 있다.

### ESLint

ESLint는 자바스크립트 린트 도구로, 코드의 오류와 스타일을 검사할 수 있다. ESLint는 코드 품질을 향상시키고 일관성을 유지할 수 있게 해준다. ESLint는 다양한 규칙을 제공하며, 사용자가 직접 규칙을 추가하거나 수정할 수 있다.

### Babel

Babel은 자바스크립트 컴파일러로, 최신 자바스크립트 문법을 하위 버전으로 변환할 수 있다. Babel은 ES6 이상의 문법을 ES5로 변환하여 모든 브라우저에서 동작할 수 있게 해준다. Babel은 플러그인을 사용하여 사용자가 원하는 기능을 추가할 수 있다.

## Vue.js 시작하기

### 프로젝트 생성

