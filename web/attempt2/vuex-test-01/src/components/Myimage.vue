<template>
  <div class="container">
    <img width='100' height='100' :src="$store.state.gifurl"/>
  </div>
</template>


<script>
  import { mapGetters, mapActions } from 'vuex'

  export default {
    name: 'hello',
    data () {
      return {
        msg: 'Welcome to Your Vue.js App'
      }
    }
  }
</script>
