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
    radiant_transformer: TeamTransformer
    dire_transformer: TeamTransformer
    match_mlp: eqx.nn.MLP


    def __init__(self, embedding_dim: int, key: jax.random.PRNGKey):
        self.embedding = jax.random.normal(key, (NUM_HEROS, embedding_dim,))



    def forward(self, x):

        hero_ids = jax.vmap(jax.nn.one_hot,()) (x[:10], NUM_HEROS) # expect the first 10 entries to be the hero ID
        # check if that has shape 10, 145
        embedded_hero_ids =  hero_ids @ self.embedding 
        # check, that that has shape (10, embedding_dim)
        player1 = jnp.concatenate(embedded_hero_ids[0], x[...]) # insert correct location in x
        player2 = jnp.concatenate(embedded_hero_ids[1], x[...]) # insert correct location in x
        player3 = jnp.concatenate(embedded_hero_ids[2], x[...]) # insert correct location in x
        player4 = jnp.concatenate(embedded_hero_ids[3], x[...]) # insert correct location in x
        player5 = jnp.concatenate(embedded_hero_ids[4], x[...]) # insert correct location in x
        player6 = jnp.concatenate(embedded_hero_ids[5], x[...]) # insert correct location in x
        player7 = jnp.concatenate(embedded_hero_ids[6], x[...]) # insert correct location in x
        player8 = jnp.concatenate(embedded_hero_ids[7], x[...]) # insert correct location in x
        player9 = jnp.concatenate(embedded_hero_ids[8], x[...]) # insert correct location in x
        player10 = jnp.concatenate(embedded_hero_ids[9], x[...]) # insert correct location in x

        playerdata = jnp.stack([player1, player2, player3, player4, player5, player6, player7, player8, player9, player10], axis=0)
        # check for dims 10, ...

        playerdata = jax.vmap(self.player_mlp, in_axes=0)(playerdata) 
        # check for dims 10, ...

        # hier jetzt transformer auf playerdata[:5] und transformer auf playerdata[5:] anwenden (gucken, das rasdiant wirklich erste 5) !! jeweils gleicher transformer der angewendet wird (aber halt einmal auf team radiant und einmal auf team dire!!)
        # -> gucken, dass dabei der input wirklich 5 verschiedene vektoren sind (die dann vom transformer embedded werden) und nicht nur eine flattete sequenz weil wir wollen ja vorallem attention zwischen den spielern und nicht zwischen jedem element (nochmal überlegen, ob das so richtig wenn ich mich wieder richtig mit transformern auseinandergesetzt habe)
        # nur 2 self-attention blöcke und nicht mehr (vllt sogar nur einer weil aktuell nicht so viele daten)

        # dann match_mlp auf die reste von x, also x[C:] wo C halt den Punkt angibt wo playerdaten aufhören (daten angucken, vllt sind es auch die ersten einträge)

        # am ende sigma-fkt anwenden s.d. output zwischen 0 und 1 (und dann nochmal schauen wie man für classification trainiert) - bei inference einfach gucken ob >/< als 0.5


    
