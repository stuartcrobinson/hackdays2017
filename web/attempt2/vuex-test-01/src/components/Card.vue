<template>
  <div style="cursor:pointer" @click="theclick(stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object)">
    <img width="150" :src="imgUrl"/>
    <br>
    <strong>{{ productTitle }}</strong>
    <br>
    {{ productId }}
  </div>
</template>

<script>
  export default {
    props: ['imgUrl', 'productTitle', 'productDescription', 'productId', 'theclick', 'stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object']
  }
</script>
