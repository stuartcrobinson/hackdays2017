<template>
  <div id="example1">
    {{ msg }} nickname: {{ nickname }}
  </div>
</template>

<script>
  export default {
    props: {
      nickname: {
        default: 'ned',
        type: String
      }
    },
    data () {
      return {
        msg: 'Welcome to Your Vue.js App'
      }
    }
  }
</script>
