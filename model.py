import jax
from jax import numpy as jnp
import equinox as eqx

####################
# CONFIGURATION
####################

NUM_HEROS = 145  # we expect 145 heroes in the game, hence hero_vector will be one-hot-encoded with 145 dimensions


####################
# CLASS DEFINITIONS
####################

class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, d_model, key):
        key_attn, key_mlp = jax.random.split(key)

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=2,
            query_size=d_model,
            key=key_attn,
        )

        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

        self.mlp = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=2 * d_model,
            depth=2,
            key=key_mlp,
        )

    def __call__(self, x):
        # x: (5, d_model)
        attended = self.attention(x, x, x)
        x = jax.vmap(self.norm1)(x + attended)

        mlp_out = jax.vmap(self.mlp)(x)
        x = jax.vmap(self.norm2)(x + mlp_out)

        return x


class TeamTransformer(eqx.Module):
    projection: eqx.nn.Linear
    block: TransformerBlock
    d_model: int = eqx.field(static=True)

    def __init__(self, n_features, key, d_model=32):
        key_proj, key_block = jax.random.split(key)

        self.projection = eqx.nn.Linear(
            n_features, d_model, key=key_proj
        )
        self.block = TransformerBlock(d_model, key_block)
        self.d_model = d_model

    def __call__(self, players):
        # players: (5, n_features)
        x = jax.vmap(self.projection)(players)
        x = self.block(x)

        # Permutationsinvariante Team-Repräsentation
        return jnp.mean(x, axis=0)

class Model(eqx.Module):
    embedding: jnp.ndarray
    player_mlp: eqx.nn.MLP
    """
    radiant_transformer: TeamTransformer
    dire_transformer: TeamTransformer
    match_mlp: eqx.nn.MLP
    """


    def __init__(self, embedding_dim: int, key: jax.random.PRNGKey):
        key_embedding, key_player = jax.random.split(key)        
        self.embedding = jax.random.normal(key_embedding, (NUM_HEROS, embedding_dim,))
        self.player_mlp = eqx.nn.MLP(in_size=15, out_size=32, width_size=32, depth=2, activation=jax.nn.relu, key=key_player)



    def  __call__(self, x):

        # embedding the hero ids
        hero_ids = jax.nn.one_hot(x[4::8], NUM_HEROS) 
        embedded_hero_ids =  hero_ids @ self.embedding 

        # connecting hero id embedding with stats for every player
        player1 = jnp.concatenate((embedded_hero_ids[0], x[4:11]), axis=0) 
        player2 = jnp.concatenate((embedded_hero_ids[1], x[12:19]), axis=0) 
        player3 = jnp.concatenate((embedded_hero_ids[2], x[19:26]), axis=0) 
        player4 = jnp.concatenate((embedded_hero_ids[3], x[27:34]), axis=0) 
        player5 = jnp.concatenate((embedded_hero_ids[4], x[35:42]), axis=0) 
        player6 = jnp.concatenate((embedded_hero_ids[5], x[43:50]), axis=0) 
        player7 = jnp.concatenate((embedded_hero_ids[6], x[51:58]), axis=0) 
        player8 = jnp.concatenate((embedded_hero_ids[7], x[59:66]), axis=0)
        player9 = jnp.concatenate((embedded_hero_ids[8], x[67:74]), axis=0) 
        player10 = jnp.concatenate((embedded_hero_ids[9], x[75:82]), axis=0)  

        # stack playerdata
        playerdata = jnp.stack([player1, player2, player3, player4, player5, player6, player7, player8, player9, player10], axis=0)

        # apply player_mlp to each player
        playerdata = jax.vmap(self.player_mlp, in_axes=0)(playerdata) 

        # BEVOR ICH HIER JETZT TRANSFORMER EINFÜHRE EINMAL NOCH NEN TUTORIAL DAZU SCHAUEN
        """
        # hier jetzt transformer auf playerdata[:5] und transformer auf playerdata[5:] anwenden (gucken, das rasdiant wirklich erste 5) !! jeweils gleicher transformer der angewendet wird (aber halt einmal auf team radiant und einmal auf team dire!!)
        # -> gucken, dass dabei der input wirklich 5 verschiedene vektoren sind (die dann vom transformer embedded werden) und nicht nur eine flattete sequenz weil wir wollen ja vorallem attention zwischen den spielern und nicht zwischen jedem element (nochmal überlegen, ob das so richtig wenn ich mich wieder richtig mit transformern auseinandergesetzt habe)
        # nur 2 self-attention blöcke und nicht mehr (vllt sogar nur einer weil aktuell nicht so viele daten)

        # dann match_mlp auf die reste von x, also x[C:] wo C halt den Punkt angibt wo playerdaten aufhören (daten angucken, vllt sind es auch die ersten einträge)

        # am ende sigma-fkt anwenden s.d. output zwischen 0 und 1 (und dann nochmal schauen wie man für classification trainiert) - bei inference einfach gucken ob >/< als 0.5
        """


if __name__ == "__main__":
    # testing if the model works on a single input-output-pair 
    testarray = jnp.ones(90)
    matchinfo = testarray[:5]
    output = testarray[6]
    input = testarray[6:]

    testkey = jax.random.key(42)
    model = Model(embedding_dim=8, key = testkey)
    output = model(input)
    

