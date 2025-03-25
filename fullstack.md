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
- npm create vue@3.10.4
- Project name: vue-project
- Add TypeScript: No (타입스크립트 사용여부 선택)
- Add JSX Support: No (JSX 사용여부 선택 jsx란? 자바스크립트의 확장 문법으로, HTML과 자바스크립트를 함께 사용할 수 있다.)
- Add vue Router for single page app: No (싱글 페이지 앱을 위한 vue Router 추가 여부 선택 router란? 웹 페이지 간의 이동을 관리하는 라이브러리)
- Add Vitest for unit testing: No (유닛 테스트를 위한 Vitest 추가 여부 선택 vitest란? Vue.js 애플리케이션을 테스트하기 위한 라이브러리)
- Add an End-to-End testing solution: No (E2E 테스트 솔루션 추가 여부 선택 e2e란? 애플리케이션의 전체적인 흐름을 테스트하는 방법)

- cd vue-project
- npm install : 프로젝트에 필요한 패키지 설치
- npm run dev : 개발 서버 실행

### Vue.js 구조
```
프로젝트 루트
├── public
│   ├── index.html       # 애플리케이션의 기본 HTML 파일
│   └── 기타 정적 파일   # favicon 등 정적 리소스
├── src
│   ├── main.js          # 애플리케이션의 진입점 파일
│   ├── App.vue          # 루트 컴포넌트
│   ├── components       # 재사용 가능한 Vue 컴포넌트 폴더
│   ├── assets           # 이미지, CSS 등 정적 파일
│   ├── router           # Vue Router 설정 폴더
│   ├── views            # 페이지 단위 컴포넌트 폴더
│   ├── store            # Vuex 상태 관리 폴더
│   ├── plugins          # 플러그인 설정 폴더
│   ├── utils            # 유틸리티 함수 폴더
│   ├── api              # API 호출 관련 파일 폴더
│   ├── styles           # 전역 스타일 파일 폴더
│   └── mixins           # 재사용 가능한 믹스인 폴더
├── node_modules         # 프로젝트 의존성 모듈 (자동 생성)
├── package.json         # 프로젝트 설정 및 스크립트 정의
├── package-lock.json    # 의존성 버전 고정 파일
└── README.md            # 프로젝트 설명 파일
```

### 주요 폴더 설명
1. **`public`**:
   - 정적 파일을 저장하는 폴더.
   - `index.html`은 Vue 앱이 렌더링될 HTML 템플릿.

2. **`src`**:
   - 애플리케이션의 핵심 코드가 위치.
   - 주요 하위 폴더:
     - `components`: 재사용 가능한 Vue 컴포넌트.
     - `assets`: 이미지, CSS 등 정적 리소스.
     - `router`: Vue Router 설정 파일.
     - `views`: 페이지 단위 컴포넌트.
     - `store`: Vuex 상태 관리 관련 파일.
     - `plugins`: 플러그인 초기화 코드.
     - `utils`: 유틸리티 함수 모음.
     - `api`: API 호출 관련 코드.
     - `styles`: 전역 스타일 파일.
     - `mixins`: 재사용 가능한 Vue 믹스인.

3. **`node_modules`**:
   - `npm install`로 설치된 의존성 모듈.

4. **`package.json`**:
   - 프로젝트 설정, 의존성, 스크립트 정의.

5. **`package-lock.json`**:
   - 의존성 버전을 고정하여 일관성을 유지.

6. **`README.md`**:
   - 프로젝트 설명 및 사용법 문서.


### 컴포넌트

- **SFC(Single File Component)**: Vue.js에서 사용하는 컴포넌트 파일 형식으로, 템플릿, 스크립트, 스타일을 한 파일에 작성할 수 있다.
- Template: HTML 태그로 구성된 마크업을 작성한다.
- Script: Vue 인스턴스와 데이터, 메소드를 정의한다.
- Style: 컴포넌트에만 적용되는 CSS 스타일을 작성한다.

```vue
<!-- HelloWorld.vue -->
<template>
  <div>
    <h1>{{ message }}</h1>
    <button @click="changeMessage">변경</button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      message: 'Hello Vue!'
    }
  },
  methods: {
    changeMessage() {
      this.message = 'Message changed!'
    }
  }
}
</script>

<style scoped>
h1 {
  color: #42b983;
}
</style>
```

## Setup Hook

셋업훅이란? 컴포넌트가 생성될 때 실행되는 훅으로 컴포지션 api의 핵심함수이다. 데이터를 설정하고 로직을 정의하는 일을 함

- hook : Vue 컴포넌트 라이프사이클의 특정 시점에 실행되는 함수들로, created, mounted, updated, destroyed 등이 있음

```vue
import { 
  onBeforeMount, 
  onMounted, 
  onBeforeUpdate, 
  onUpdated, 
  onBeforeUnmount, 
  onUnmounted 
} from 'vue';

// <script setup> 내부에서 사용
onBeforeMount(() => {
  console.log('마운트 전');
});

onMounted(() => {
  console.log('마운트 됨');
});

onBeforeUpdate(() => {
  console.log('업데이트 전');
});

onUpdated(() => {
  console.log('업데이트 됨');
});

onBeforeUnmount(() => {
  console.log('언마운트 전');
});

onUnmounted(() => {
  console.log('언마운트 됨');
});
```

- composition api : Vue 3에서 도입된 새로운 API 방식으로, 기존의 Options API와 달리 관련 기능을 함께 그룹화하여 코드의 재사용성과 가독성을 높이는 방식

- setup : Composition API의 진입점 함수로, 컴포넌트가 생성되기 전에 실행되며 반응형 데이터, 계산된 속성, 메서드, 라이프사이클 훅 등을 정의할 수 있음

```vue
<template>
  <div>
    <p>Count: {{ count }}</p>
    <button @click="increment">증가</button>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

// 반응형 상태 정의
const count = ref(0)

// 메서드 정의
function increment() {
  count.value++
}

// 라이프사이클 훅 사용
onMounted(() => {
  console.log('컴포넌트가 마운트되었습니다')
})
// 별도의 return 문이 필요 없음 - 모든 변수는 자동으로 템플릿에 노출됨
</script>

<style scoped>
p {
  color: #42b983;
  font-weight: bold;
}
button {
  margin-top: 10px;
  padding: 5px 10px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style>
```

`<script setup>`의 장점:
- 더 적은 보일러플레이트 코드로 간결한 작성 가능
- 변수와 함수를 직접 정의하여 템플릿에서 바로 사용 가능
- 더 나은 TypeScript 지원과 컴파일 시 성능 최적화


## 템플릿 문법

### 머스태치 문법

머스태치 문법(이중 중괄호)은 Vue.js에서 텍스트 데이터를 화면에 표시하는 가장 기본적인 방법이다.

```vue
<template>
  <div>
    <p>메시지: {{ message }}</p>
    <p>계산: {{ count * 2 }}</p>
    <p>상태: {{ isActive ? '활성' : '비활성' }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const message = ref('안녕하세요')
const count = ref(5)
const isActive = ref(true)
</script>
```

### 속성 바인딩

`v-bind` 디렉티브 또는 단축 문법 `:`을 사용해 HTML 속성에 동적으로 값을 바인딩한다.

```vue
<template>
  <div>
    <img v-bind:src="imageUrl" alt="이미지">
    <!-- 단축 문법 -->
    <a :href="linkUrl" :target="linkTarget">링크</a>
    <!-- 불리언 속성 -->
    <input type="checkbox" :disabled="isDisabled">
  </div>
</template>

<script setup>
import { ref } from 'vue'

const imageUrl = ref('/images/logo.png')
const linkUrl = ref('https://example.com')
const linkTarget = ref('_blank')
const isDisabled = ref(false)
</script>
```

### 클래스 바인딩

클래스를 동적으로 적용하기 위한 특별한 바인딩 기능을 제공한다.

```vue
<template>
  <div>
    <!-- 객체 구문 -->
    <div :class="{ active: isActive, error: hasError }">
      객체 구문 예시
    </div>
    
    <!-- 배열 구문 -->
    <div :class="[activeClass, errorClass]">
      배열 구문 예시
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const isActive = ref(true)
const hasError = ref(false)
const activeClass = ref('active')
const errorClass = ref('text-danger')
</script>
```

### 스타일 바인딩

인라인 스타일을 동적으로 바인딩할 수 있다.

```vue
<template>
  <div>
    <!-- 객체 구문 -->
    <div :style="{ color: textColor, fontSize: fontSize + 'px' }">
      스타일 객체 구문
    </div>
    
    <!-- 스타일 객체 바인딩 -->
    <div :style="styleObject">스타일 객체</div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'

const textColor = ref('red')
const fontSize = ref(16)

const styleObject = reactive({
  color: 'green',
  fontWeight: 'bold'
})
</script>
```

### 이벤트 디렉티브

`v-on` 디렉티브 또는 단축 문법 `@`을 사용해 DOM 이벤트를 처리한다.

```vue
<template>
  <div>
    <!-- 기본 이벤트 처리 -->
    <button v-on:click="increment">증가</button>
    
    <!-- 단축 문법 -->
    <button @click="decrement">감소</button>
    
    <!-- 이벤트 수식어 -->
    <form @submit.prevent="submitForm">
      <button type="submit">제출</button>
    </form>
    
    <!-- 키 수식어 -->
    <input @keyup.enter="processInput">
  </div>
</template>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
  count.value++
}

function decrement() {
  count.value--
}

function submitForm() {
  console.log('폼 제출됨')
}

function processInput(event) {
  console.log('입력:', event.target.value)
}
</script>
```

주요 이벤트 수식어:
- `.stop`: 이벤트 전파 중단
- `.prevent`: 기본 동작 방지
- `.once`: 이벤트를 한 번만 처리
- `.self`: 이벤트가 해당 요소에서 직접 발생한 경우에만 처리

주요 키 수식어:
- `.enter`, `.tab`, `.delete`, `.esc`, `.space`, `.up`, `.down`

### 반복 디렉티브

`v-for` 디렉티브는 배열이나 객체의 데이터를 기반으로 항목을 반복해서 렌더링한다.

```vue
<template>
  <div>
    <!-- 배열 반복 -->
    <ul>
      <li v-for="(item, index) in items" :key="item.id">
        {{ index }}. {{ item.name }}
      </li>
    </ul>
    
    <!-- 객체 속성 반복 -->
    <div v-for="(value, key, index) in user" :key="key">
      {{ index }}. {{ key }}: {{ value }}
    </div>
    
    <!-- 범위 반복 -->
    <span v-for="n in 5" :key="n">{{ n }} </span>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const items = ref([
  { id: 1, name: '사과' },
  { id: 2, name: '바나나' },
  { id: 3, name: '오렌지' }
])

const user = ref({
  name: '홍길동',
  age: 30,
  job: '개발자'
})
</script>
```

#### 주요 특징

1. **`:key` 속성**: 
   - 각 항목에 고유한 키를 제공해 Vue가 DOM을 효율적으로 업데이트할 수 있게 한다.
   - 가능하면 항상 `:key`를 사용해야 한다.

2. **구조 분해 할당**:
   ```vue
   <li v-for="{ name, id } in items" :key="id">
     {{ name }}
   </li>
   ```

3. **배열 변경 감지**:
   - `push()`, `pop()`, `shift()`, `unshift()`, `splice()`, `sort()`, `reverse()` 등의 배열 메서드는 자동으로 화면 업데이트를 트리거한다.

4. **필터링 및 정렬**:
   ```vue
   <li v-for="item in filteredItems" :key="item.id">
     {{ item.name }}
   </li>
   ```

   ```js
   const filteredItems = computed(() => {
     return items.value.filter(item => item.price > 100)
   })
   ```

5. **v-for와 v-if 함께 사용**:
   - 같은 요소에 v-for와 v-if를 함께 사용하지 않는 것이 좋다.
   - 대신 computed 속성을 사용하거나 wrapper 요소를 추가하는 것이 권장된다.

   ```vue
   <!-- 권장하지 않음 -->
   <li v-for="item in items" v-if="item.isActive" :key="item.id">...</li>
   
   <!-- 대신 computed 속성 사용 -->
   <li v-for="item in activeItems" :key="item.id">...</li>
   ```

   ```js
   const activeItems = computed(() => {
     return items.value.filter(item => item.isActive)
   })
   ```

반복 디렉티브는 목록, 테이블, 그리드 등의 데이터 기반 UI 요소를 쉽게 구현할 수 있게 해준다.

## 반응형 상태


