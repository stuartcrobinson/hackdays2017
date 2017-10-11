import Vue from 'vue'
import VueRouter from 'vue-router'
import Hello from '@/components/Hello'
import Example1 from '@/components/Example1'
import Example2 from '@/components/Example2'

Vue.use(VueRouter)

//
//
// var Home = Vue.extend({
//   template: 'Welcome to the <b>home page</b>!';
// });
//
// var People = Vue.extend({
//   template: 'Look at all the people who work here!';
// });
//
// router.map({
//   '/': {
//     component: Home
//   },
//   '/people': {
//     component: People
//   }
// });
//

// var router = new VueRouter()
//
// const User = {
//   template: '<div>User {{ $route.params.id }}</div>'
// }
//
//
// const router = new VueRouter({
//   routes: [
//     { path: '/user/:id', component: User }
//   ]
// })
//
// Product: {<
//   template > hi < /template>};

export default new VueRouter({
  routes: [
    {
      path: '/',
      name: 'Hello',
      component: Hello
    },
    {
      path: '/product/:productId',
      name: 'productt',
      component: {
        template: '<h1>hi {{this.$route.params.productId}}</h1>'
      }
    },
    {
      path: '/one/two/three',
      name: 'Example11',
      component: Example1
    },
    {
      path: '/lalalala',
      name: 'Example22',
      component: Example2
    }
  ]
})
