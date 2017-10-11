<template>
  <div class="container0">
    <div>
      <h1>{{ msg }}</h1>
      <h2>Essential Links for  {{ $store.state.name }}  {{ $store.state.count }}</h2>
      Counter: {{ $store.state.count }} times, count is {{ evenOrOdd }}.
      <br>
      Answer: {{ $store.state.isyes }}
      <br>
      gifurl: {{ $store.state.gifurl }}
      <br>
      <img :src="$store.state.gifurl"/>
      <!--{{ $store.state.gifurl }}-->
      <ul>
        <li><a href="https://vuejs.org" target="_blank">Core Docs</a></li>
        <li><a href="https://forum.vuejs.org" target="_blank">Forum</a></li>
        <li><a href="https://chat.vuejs.org" target="_blank">Community Chat</a></li>
        <li><a href="https://twitter.com/vuejs" target="_blank">Twitter</a></li>
        <li><a href="https://www.google.com" target="_blank">Google</a></li>

        <br>
        <li><a href="http://vuejs-templates.github.io/webpack/" target="_blank">Docs for This Template</a></li>
      </ul>
      <h2>Ecosystem</h2>
      <ul>
        <li><a href="http://router.vuejs.org/" target="_blank">vue-router</a></li>
        <li><a href="http://vuex.vuejs.org/" target="_blank">vuex</a></li>
        <li><a href="http://vue-loader.vuejs.org/" target="_blank">vue-loader</a></li>
        <li><a href="https://github.com/vuejs/awesome-vue" target="_blank">awesome-vue</a></li>
      </ul>
    </div>
    <div class="container">
      <div class="row text-center">
        <h3>Clicked: {{ $store.state.count }} times, count is {{ evenOrOdd }}.</h3>
        <button class="btn btn-success" @click="increment">+</button>
        <button class="btn btn-danger" @click="decrement">-</button>
        <button class="btn" @click="incrementIfOdd">Increment if odd</button>
        <button class="btn" @click="incrementAsync">Increment async</button>
        <button class="btn" @click="yesnoaction">yesnoaction</button>
        <button class="btn" @click="myfunction">load products {{ namee }}</button>
        <br>
        <!--<form id="demo">-->
        <form @submit.prevent="myfunction">
          <label for="nameId">Enter name:</label>
          <input type="text" v-model="namee" id="nameId" lazy/>
        </form>

        <p>{{ namee }} is {{ age }} years old.</p>
        <br>
        is loading? {{ $store.state.isLoadingProductsSearch }}
        <!--<br>-->
        <!--products? {{ $store.state.queriedProducts }}-->
        <br>
        length? {{ $store.state.queriedProducts.length }}
        <br>
        <ol>
          <li>one</li>
          <li>two</li>
          <li>three</li>
        </ol>
        <ol id="example-1">
          <li v-for="item in $store.state.queriedProducts">

            <!--<card :productTitle="item._source.title"></card>-->

            <!--<br>-->
            <!--{{item}}-->
            <!--<br>-->
            <card :imgUrl="item._source.imageUrl"
                  :productTitle="item._source.title"
                  :productDescription="item._source.description"
                  :productId="item._source.productId"></card>
            <!--{{ item }}-->
          </li>
        </ol>
      </div>
    </div>
    <example1 nickname="bob">hi</example1>
  </div>
</template>

<!--<script>-->
<!--import { mapGetters, mapActions } from 'vuex'-->

<!--export default {-->
<!--name: 'hello',-->
<!--computed: mapGetters([-->
<!--'evenOrOdd'-->
<!--]),-->
<!--methods: mapActions([-->
<!--'increment',-->
<!--'decrement',-->
<!--'incrementIfOdd',-->
<!--'incrementAsync'-->
<!--])-->
<!--}-->
<!--</script>-->

<script>
  import { mapGetters, mapActions } from 'vuex'
  import Example1 from './Example1.vue'
  import Card from './Card.vue'

  // https://github.com/vuejs/vuex/issues/367
  // this.$store.dispatch('updatecustomer', this.custumer)

  export default {
    name: 'hello',
    components: {
      Example1, Card
    },
    computed: mapGetters([
      'evenOrOdd'
    ]),
    methods: {
      ...mapActions([
        'increment',
        'decrement',
        'incrementIfOdd',
        'incrementAsync',
        'yesnoaction',
        'loadproducts'
      ]),
      myfunction () {
        this.$store.dispatch('loadproducts', this.namee)
      }
    },
    data () {
      return {
        msg: 'Welcome to Your Vue.js App',
        namee: 'Ashley',
        age: 100
      }
    }
  }
</script>

<!--&lt;!&ndash; Add "scoped" attribute to limit CSS to this component only &ndash;&gt;-->
<!--<style scoped>-->
<!--h1, h2 {-->
<!--font-weight: normal;-->
<!--}-->

<!--/*ul {*/-->
<!--/*list-style-type: none;*/-->
<!--/*padding: 0;*/-->
<!--/*}*/-->

<!--/*li {*/-->
<!--/*display: inline-block;*/-->
<!--/*margin: 0 10px;*/-->
<!--/*}*/-->

<!--a {-->
<!--color: #42b983;-->
<!--}-->
<!--</style>-->
