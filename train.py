import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

labels = ["contradiction", "entailment", "neutral"]
batch_size=128
epochs=5
max_length =125
x_train = pd.read_csv("./snli_1.0_train.csv", nrows=100000)
x_val = pd.read_csv("./snli_1.0_dev.csv")
x_test = pd.read_csv("./snli_1.0_test.csv")
x_train.dropna(axis=0, inplace=True)

x_train = (
    x_train[x_train.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)
x_val = (
    x_val[x_val.similarity != "-"]
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

print(f"Total train samples : {x_train.shape[0]}")
print(f"Total validation samples: {x_val.shape[0]}")
print(f"Total test samples: {x_val.shape[0]}")
print(x_train.similarity.value_counts())

x_train["label"] = x_train["similarity"].apply(lambda x: 0 if x == 
"contradiction" else 1 if x == "entailment" else 2)
x_val["label"] = x_val["similarity"].apply(lambda x: 0 if x == 
"contradiction" else 1 if x == "entailment" else 2)
x_test["label"] = x_test["similarity"].apply(lambda x: 0 if x == 
"contradiction" else 1 if x == "entailment" else 2)

y_train = tf.keras.utils.to_categorical(x_train.label, num_classes=3)
y_val = tf.keras.utils.to_categorical(x_val.label, num_classes=3)
y_test = tf.keras.utils.to_categorical(x_test.label, num_classes=3)

class DataLoader(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentences: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_labels: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_labels=False`)
    """

    def __init__(
        self,
        sentences,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_labels=True,
    ):
        self.sentences = sentences
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_labels = include_labels
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.indexes = np.arange(len(self.sentences))
        self.shuffle_data()

    def __len__(self):
        return len(self.sentences) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentences = self.sentences[indexes]
        encoded = self.tokenizer.batch_encode_plus(
            sentences.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        sentence_inputs = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_labels:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [sentence_inputs, attention_masks, token_type_ids], labels
        else:
            return [sentence_inputs, attention_masks, token_type_ids]

    def shuffle_data(self):
        if self.shuffle:
            np.random.RandomState(50).shuffle(self.indexes)


sentence_inputs = tf.keras.layers.Input(shape=(max_length,), 
dtype=tf.int32, name="input_ids")
attention_masks = tf.keras.layers.Input(shape=(max_length,), 
dtype=tf.int32, name="attention_masks")
token_type_ids = tf.keras.layers.Input(shape=(max_length,), 
dtype=tf.int32, name="token_type_ids")

bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
bert_model.trainable = True
bert_output = bert_model.bert(sentence_inputs, 
attention_mask=attention_masks, token_type_ids=token_type_ids)
sequence_output = bert_output.last_hidden_state

# pooled_output = bert_output.pooler_output
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True))(sequence_output)
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm)
x = tf.keras.layers.concatenate([avg_pool, max_pool])
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(3, activation="softmax")(x)
model = tf.keras.models.Model(inputs=[sentence_inputs, attention_masks,token_type_ids], outputs=out)


train_data = DataLoader(
    x_train[["sentence1", "sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = DataLoader(
    x_val[["sentence1", "sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)
test_data = DataLoader(
    x_test[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)

my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=15),
    tf.keras.callbacks.ModelCheckpoint(
        
filepath='/content/Sematic-loss_{loss:.3f}_val_acc{accuracy:.3f}.h5',
        monitor = 'accuracy',
        save_best_only = True
        ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = 'loss',
        factor = 0.1,
        patience = 3,
        min_lr = 1e-6
    ),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=5,
    use_multiprocessing=True,
    workers=-1,
)

model.save("./my_model.h5")
