import os

import tensorflow as tf


def abs_path_file(filename):
    return os.path.abspath(filename)


class LoadedModelFromSession:
    def __init__(self, session):
        self.session = session
        self.graph = session.graph

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.session.run(fetches, feed_dict, options, run_metadata)

    def close(self):
        tf.compat.v1.reset_default_graph()
        self.session.close()


class LoadedModelFromSavedModelPB(LoadedModelFromSession):
    def __init__(self, saved_model_file):
        self.graph = tf.Graph()
        session = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.saved_model.loader.load(
            session,
            [tf.saved_model.TRAINING],
            saved_model_file)
        super().__init__(session)


class LoadedModelFromMetaAndCheckpoint(LoadedModelFromSession):
    def __init__(self, meta_file, checkpoint):
        tf.compat.v1.reset_default_graph()
        saver = tf.compat.v1.train.import_meta_graph(abs_path_file(meta_file))
        self.graph = tf.compat.v1.get_default_graph()
        session = tf.compat.v1.Session(graph=self.graph)
        checkpoint = tf.train.latest_checkpoint(abs_path_file(checkpoint))
        saver.restore(session, checkpoint)
        super().__init__(session)

class LoadedModelFromGraphDef(LoadedModelFromSession):
    def __init__(self, graphdef):
        tf.compat.v1.reset_default_graph()
        tf.import_graph_def(graphdef)
        self.graph = tf.compat.v1.get_default_graph()
        session = tf.compat.v1.Session(graph=self.graph)
        super().__init__(session)
