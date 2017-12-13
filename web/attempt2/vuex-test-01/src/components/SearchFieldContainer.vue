<template>
  <div>
    <vue-datalist-autocomplete placeholder="search High Fashion Home catalog" :onsubmit="queryProductsMethod" typeahead_endpoint_partial="/api/queryproducttitletypeaheads/"/>
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
        console.log('SUBMITTED queryProductsMethod query:' + query)
//        this.$store.dispatch('search_catalog', query)
        // search_catalogExpress
        this.$store.dispatch('search_catalogExpress', query)
      }
    }
  }
</script>

<!--TODO - clear out previous result images when submit new query-->
