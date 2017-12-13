<template>
  <div id="app">

    <div id="some example stuff" style="display: none">
      <example1 nickname="sally">example1</example1>
      <!--<router-link :to="{ name: 'user', params: { userId: 123 }}">User</router-link>-->
      <router-link :to="{ name: 'Example11'}">Example11</router-link>
      <router-link :to="{ name: 'Example22'}">Example22</router-link>
      <router-link :to="{ name: 'Hello'}">Hello</router-link>

      <router-link to="Example11">Example11 doesntwork</router-link>
      <router-link to="Example22">Example22 doesntwork</router-link>
      <router-link to="/">Home</router-link>

      <!--<img src="./assets/logo.png">-->
    </div>
    <router-view></router-view>
  </div>
</template>

<script>
  import Example1 from './components/Example1.vue'

  export default {

    components: {
      Example1
    },
    name: 'app'
  }
</script>

<style>
  #app {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 60px;
  }
</style>
