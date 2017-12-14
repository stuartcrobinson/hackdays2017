// https://medium.com/codingthesmartway-com-blog/vue-js-2-state-management-with-vuex-introduction-db26cb495113
// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import Example1 from './components/Example1.vue'
// You need a specific loader for CSS files like https://github.com/webpack/css-loader

import store from './store/store'

Vue.config.productionTip = false

document.title = 'Hackdays2017'

// Vue.component('example1', Example1)

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  store,
  template: '<App/>',
  components: {App, Example1}
})
