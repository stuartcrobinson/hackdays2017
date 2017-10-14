<template>
  <div>
    <vue-datalist-autocomplete :onsubmit="queryProductsMethod" typeahead_endpoint_partial="/api/queryproducttitletypeaheads/"/>
  </div>
</template>

<script>
  import { mapActions } from 'vuex'
  import VueDatalistAutocomplete from './VueDatalistAutocomplete.vue'

  export default {
    components: {
      VueDatalistAutocomplete
    },
    methods: {
      ...mapActions([
        'loadproducts'
      ]),
      queryProductsMethod (query) {
        this.$store.dispatch('loadproducts', query)
      }
    }
  }
</script>

<!--TODO - clear out previous result images when submit new query-->
