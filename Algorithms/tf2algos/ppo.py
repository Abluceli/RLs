import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class PPO(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=128,
                 base_dir=None,

                 epoch=5,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 share_net=True,
                 logger2file=False,
                 out_graph=False):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            gamma=gamma,
            max_episode=max_episode,
            base_dir=base_dir,
            policy_mode='ON',
            batch_size=batch_size)
        self.lr = lr
        self.beta = beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.share_net = share_net
        self.sigma_offset = np.full([self.a_counts, ], 0.01)
        if self.action_type == 'continuous':
            self.net = Nn.a_c_v_continuous(self.s_dim, self.visual_dim, self.a_counts, 'ppo_net')
            self.actor_net = Nn.actor_continuous(self.s_dim, self.visual_dim, self.a_counts, 'actor_net')
        else:
            self.net = Nn.a_c_v_discrete(self.s_dim, self.visual_dim, self.a_counts, 'ppo_net')
            self.actor_net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'actor_net')
        self.critic_net = Nn.critic_v(self.s_dim, self.visual_dim, 'critic_net')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘ　　
        ''')

    def choose_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self._get_action(s, visual_s).numpy()
        else:
            if np.random.uniform() < self.epsilon:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self._get_action(s, visual_s).numpy()
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                if self.share_net:
                    mu, sigma, _ = self.net(vector_input, visual_input)
                else:
                    mu, sigma = self.actor_net(vector_input, visual_input)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
            else:
                if self.share_net:
                    log_action_probs, _ = self.net(vector_input, visual_input)
                else:
                    log_action_probs = self.actor_net(vector_input, visual_input)
                norm_dist = tfp.distributions.Categorical(probs=tf.exp(log_action_probs))
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            'done': done,
            'value': np.squeeze(self._get_value(s, visual_s).numpy()),
            'prob': self._get_new_prob(s, visual_s, a).numpy() + 1e-10
        }, ignore_index=True)
        self.s_ = s_
        self.visual_s_ = visual_s_

    @tf.function
    def _get_value(self, s, visual_s):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                if self.share_net:
                    _, _, value = self.net(s, visual_s)
                else:
                    value = self.critic_net(s, visual_s)
            else:
                if self.share_net:
                    _, value = self.net(s, visual_s)
                else:
                    value = self.critic_net(s, visual_s)
            return value

    @tf.function
    def _get_new_prob(self, s, visual_s, a):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                if self.share_net:
                    mu, sigma, _ = self.net(s, visual_s)
                else:
                    mu, sigma = self.actor_net(s, visual_s)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                new_prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
            else:
                if self.share_net:
                    log_action_probs, _ = self.net(s, visual_s)
                else:
                    log_action_probs = self.actor_net(s, visual_s)
                new_prob = tf.reduce_sum(tf.multiply(tf.exp(log_action_probs), a), axis=1, keepdims=True)
            return new_prob

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        init_value = np.squeeze(self._get_value(self.s_, self.visual_s_).numpy())
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)
        self.data['td_error'] = sth.discounted_sum_minus(
            self.data.r.values,
            self.gamma,
            init_value,
            self.data.done.values,
            self.data.value.values
        )
        # GAE
        self.data['advantage'] = sth.discounted_sum(
            self.data.td_error.values,
            self.lambda_ * self.gamma,
            0,
            self.data.done.values
        )
        # self.data.to_excel(self.excel_writer, sheet_name='test', index=True)
        # self.excel_writer.save()

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack(i_data.s.values)
        visual_s = np.vstack(i_data.visual_s.values)
        a = np.vstack(i_data.a.values)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1)
        old_prob = np.vstack(i_data.prob.values)
        advantage = np.vstack(i_data.advantage.values).reshape(-1, 1)
        return s, visual_s, a, dc_r, old_prob, advantage

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, visual_s, a, dc_r, old_prob, advantage = self.get_sample_data()
            if self.share_net:
                actor_loss, critic_loss, entropy = self.train_share(s, visual_s, a, dc_r, old_prob, advantage)
            else:
                actor_loss, critic_loss, entropy = self.train_not_share(s, visual_s, a, dc_r, old_prob, advantage)
        tf.summary.experimental.set_step(episode)
        tf.summary.scalar('LOSS/entropy', entropy)
        tf.summary.scalar('LOSS/actor_loss', actor_loss)
        tf.summary.scalar('LOSS/critic_loss', critic_loss)
        tf.summary.scalar('LEARNING_RATE/lr', self.lr)
        tf.summary.scalar('REWARD/discounted_reward', self.data.discounted_reward.values[0].mean())
        tf.summary.scalar('REWARD/total_reward', self.data.total_reward.values[0].mean())
        self.recorder.writer.flush()
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, s, visual_s, a, dc_r, old_prob, advantage):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu, sigma, value = self.net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    new_prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    log_action_probs, value = self.net(s, visual_s)
                    new_prob = tf.reduce_sum(tf.multiply(tf.exp(log_action_probs), a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_action_probs) * log_action_probs, axis=1, keepdims=True))
                ratio = new_prob / old_prob
                surrogate = ratio * advantage
                td_error = dc_r - value
                actor_loss = tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    ))
                value_loss = tf.reduce_mean(tf.square(td_error))
                loss = -(actor_loss - 1.0 * value_loss + self.beta * entropy)
            loss_grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, value_loss, entropy
    
    @tf.function(experimental_relax_shapes=True)
    def train_not_share(self, s, visual_s, a, dc_r, old_prob, advantage):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                value = self.critic_net(s, visual_s)
                td_error = dc_r - value
                value_loss = tf.reduce_mean(tf.square(td_error))
            critic_grads = tape.gradient(value_loss, self.critic_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu, sigma = self.actor_net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    new_prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    log_action_probs = self.actor_net(s, visual_s)
                    new_prob = tf.reduce_sum(tf.multiply(tf.exp(log_action_probs), a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(log_action_probs) * log_action_probs, axis=1, keepdims=True))
                ratio = new_prob / old_prob
                surrogate = ratio * advantage
                actor_loss = -(tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    )) + self.beta * entropy)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, value_loss, entropy