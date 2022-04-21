"""MSCOCO Semantic Segmentation pretraining for VOC."""
import os
import pickle
import torch
import numpy as np

from tqdm import trange
from PIL import Image
from .segbase import SegmentationDataset


class FoodRecognitionSegmentation(SegmentationDataset):
    """COCO Semantic Segmentation Dataset for VOC Pre-training.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/coco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    # CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
    #             1, 64, 20, 63, 7, 72]
    CAT_LIST = [50, 101246, 100546, 101129, 101243, 100133, 101306, 101126, 101305, 100206, 101178, 101150, 101185, 101166, 100966, 100078, 100107, 101181, 101335, 100523, 101291, 100063, 101219, 101308, 100089, 101183, 101279, 101188, 101311, 101214, 100321, 101275, 100319, 100838, 101302, 101347, 100184, 100182, 101282, 101144, 100175, 100130, 100145, 100083, 101355, 101194, 101284, 101314, 100790, 101165, 100022, 101354, 101172, 100456, 101236, 100710, 101197, 101180, 100180, 101294, 101193, 101138, 100843, 100352, 100082, 101043, 101022, 100486, 100929, 100161, 101265, 101361, 100674, 101260, 100196, 100245, 100318, 101210, 101208, 101148, 100102, 100235, 100093, 101248, 101358, 100249, 101255, 101326, 101215, 101292, 100963, 100077, 100243, 101258, 101027, 100156, 100152, 100080, 100925, 100960, 100072, 100338, 100687, 100495, 100563, 100826, 100172, 100757, 100150, 100752, 100652, 101310, 100962, 101177, 100099, 101176, 101317, 100942, 100176, 100883, 101212, 101262, 100836, 101340, 100993, 100060, 100300, 101254, 101285, 100108, 100949, 101168, 101297, 100974, 100069, 101187, 101322, 100195, 100333, 100325, 101272, 100302, 101268, 100335, 101009, 101182, 101156, 101220, 100355, 100224, 100594, 100264, 100185, 101346, 100142, 100937, 100086, 100111, 101360, 101200, 100978, 101324, 101201, 100442, 100229, 101296, 100067, 100859, 100645, 101363, 101213, 100445, 100421, 100070, 101190, 100537, 100183, 101141, 101338, 100115, 101298, 100234, 100301, 100171, 100607, 101114, 101247, 101256, 100064, 100157, 101325, 101199, 100477, 101135, 100916, 101349, 100131, 101147, 101329, 100951, 100192, 101123, 101149, 101328, 101341, 100911, 101295, 101121, 100360, 101331, 101273, 100141, 101229, 100982, 100084, 101257, 100092, 100658, 100123, 100101, 101343, 100059, 100834, 100324, 101309, 100725, 101012, 100334, 101304, 101038, 101218, 101238, 100332, 100105, 100140, 100190, 100250, 101244, 100073, 101307, 100143, 101303, 100842, 100475, 101189, 100314, 100200, 101209, 101170, 101014, 100164, 101173, 100129, 100177, 100349, 101237, 100068, 100057, 100423, 101159, 100958, 101271, 100719, 100178, 100076, 101327, 101231, 100348, 100907, 100049, 100467, 100895, 100135, 101153, 100158, 100181, 100840, 101323, 100957, 100173, 100742, 100992, 100681, 100179, 101029, 101240, 100310, 101164, 101290, 100146, 100346, 100649, 100315, 101249, 101118, 100844, 100926, 100909, 101283, 101196, 100437, 100781, 100104, 100282, 100132, 100936, 100337, 101032, 100848, 101006, 101228, 100364, 101266, 100322, 100174, 100114, 100186, 101077, 101274, 100031, 100113, 100151, 100160]
    NUM_CLASS = 323

    def __init__(self, root='../data', split='train', mode=None, transform=None, **kwargs):
        super(FoodRecognitionSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # lazy import pycocotools
        from pycocotools.coco import COCO
        from pycocotools import mask
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'train/annotations.json')
            ids_file = os.path.join(root, 'train/train_ids.mx')
            self.root = os.path.join(root, 'train')
        else:
            print('val set')
            ann_file = os.path.join(root, 'val/annotations.json')
            ids_file = os.path.join(root, 'val/val_ids.mx')
            self.root = os.path.join(root, 'val')
        self.coco = COCO(ann_file)
        self.coco_mask = mask

        # print("self.coco:", self.coco)
        # print("self.coco_mask:", self.coco_mask)

        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

        print("self.ids:", self.ids)

        self.transform = transform

    def __getitem__(self, index):
        # print("__getitem__ index:", index)
        coco = self.coco
        img_id = self.ids[index]
        # print("img_id:", img_id)
        img_metadata = coco.loadImgs(img_id)[0]
        # img_metadata: {'id': 165736, 'file_name': '165736.jpg', 'width': 391, 'height': 390}
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, 'images', path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        # print("os.path.basename:", os.path.basename(img_id))
        # return img, mask, os.path.basename(self.ids[index])
        return img, mask, os.path.basename(img_metadata['file_name'])

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']

            # c = cat
            if cat in self.CAT_LIST: 
                c = self.CAT_LIST.index(cat)
            else:
                continue

            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    # def _preprocess(self, ids):
    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            # print("_preprocess  img_id:", img_id, type(img_id))
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb+') as f:
            pickle.dump(new_ids, f)
        return new_ids

    def __len__(self):
        return len(self.ids)

    @property
    def classes(self):
        """Category names."""
        # return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
        #         'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        #         'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        #         'tv')

        # 323
        return ('beetroot-steamed-without-addition-of-salt', 'bread_wholemeal', 'jam', 'water', 'bread', 'banana', 'soft_cheese', 'ham_raw', 'hard_cheese', 'cottage_cheese', 'coffee', 'fruit_mixed', 'pancake', 'tea', 'salmon_smoked', 'avocado', 'spring_onion_scallion', 'ristretto_with_caffeine', 'ham_n_s', 'egg', 'bacon', 'chips_french_fries', 'juice_apple', 'chicken', 'tomato', 'broccoli', 'shrimp_prawn', 'carrot', 'chickpeas', 'french_salad_dressing', 'pasta_hornli_ch', 'sauce_cream', 'pasta_n_s', 'tomato_sauce', 'cheese_n_s', 'pear', 'cashew_nut', 'almonds', 'lentil_n_s', 'mixed_vegetables', 'peanut_butter', 'apple', 'blueberries', 'cucumber', 'yogurt', 'butter', 'mayonnaise', 'soup', 'wine_red', 'wine_white', 'green_bean_steamed_without_addition_of_salt', 'sausage', 'pizza_margherita_baked', 'salami_ch', 'mushroom', 'tart_n_s', 'rice', 'white_coffee', 'sunflower_seeds', 'bell_pepper_red_raw', 'zucchini', 'asparagus', 'tartar_sauce', 'lye_pretzel_soft', 'cucumber_pickled_ch', 'curry_vegetarian', 'soup_of_lentils_dahl_dhal', 'salmon', 'salt_cake_ch_vegetables_filled', 'orange', 'pasta_noodles', 'cream_double_cream_heavy_cream_45', 'cake_chocolate', 'pasta_spaghetti', 'black_olives', 'parmesan', 'spaetzle', 'salad_lambs_ear', 'salad_leaf_salad_green', 'potato', 'white_cabbage', 'halloumi', 'beetroot_raw', 'bread_grain', 'applesauce', 'cheese_for_raclette_ch', 'bread_white', 'curds_natural', 'quiche', 'beef_n_s', 'taboule_prepared_with_couscous', 'aubergine_eggplant', 'mozzarella', 'pasta_penne', 'lasagne_vegetable_prepared', 'mandarine', 'kiwi', 'french_beans', 'spring_roll_fried', 'caprese_salad_tomato_mozzarella', 'leaf_spinach', 'roll_of_half_white_or_white_flour_with_large_void', 'omelette_with_flour_thick_crepe_plain', 'tuna', 'dark_chocolate', 'sauce_savoury_n_s', 'raisins_dried', 'ice_tea_on_black_tea_basis', 'kaki', 'smoothie', 'crepe_with_flour_plain', 'nuggets', 'chili_con_carne_prepared', 'veggie_burger', 'chinese_cabbage', 'hamburger', 'soup_pumpkin', 'sushi', 'chestnuts_ch', 'sauce_soya', 'balsamic_salad_dressing', 'pasta_twist', 'bolognaise_sauce', 'leek', 'fajita_bread_only', 'potato_gnocchi', 'rice_noodles_vermicelli', 'bread_whole_wheat', 'onion', 'garlic', 'hummus', 'pizza_with_vegetables_baked', 'beer', 'glucose_drink_50g', 'ratatouille', 'peanut', 'cauliflower', 'green_olives', 'bread_pita', 'pasta_wholemeal', 'sauce_pesto', 'couscous', 'sauce', 'bread_toast', 'water_with_lemon_juice', 'espresso', 'egg_scrambled', 'juice_orange', 'braided_white_loaf_ch', 'emmental_cheese_ch', 'hazelnut_chocolate_spread_nutella_ovomaltine_caotina', 'tomme_ch', 'hazelnut', 'peach', 'figs', 'mashed_potatoes_prepared_with_full_fat_milk_with_butter', 'pumpkin', 'swiss_chard', 'red_cabbage_raw', 'spinach_raw', 'chicken_curry_cream_coconut_milk_curry_spices_paste', 'crunch_muesli', 'biscuit', 'meatloaf_ch', 'fresh_cheese_n_s', 'honey', 'vegetable_mix_peas_and_carrots', 'parsley', 'brownie', 'ice_cream_n_s', 'salad_dressing', 'dried_meat_n_s', 'chicken_breast', 'mixed_salad_chopped_without_sauce', 'feta', 'praline_n_s', 'walnut', 'potato_salad', 'kolhrabi', 'alfa_sprouts', 'brussel_sprouts', 'gruyere_ch', 'bulgur', 'grapes', 'chocolate_egg_small', 'cappuccino', 'crisp_bread', 'bread_black', 'rosti_n_s', 'mango', 'muesli_dry', 'spinach', 'fish_n_s', 'risotto', 'crisps_ch', 'pork_n_s', 'pomegranate', 'sweet_corn', 'flakes', 'greek_salad', 'sesame_seeds', 'bouillon', 'baked_potato', 'fennel', 'meat_n_s', 'croutons', 'bell_pepper_red_stewed', 'nuts', 'breadcrumbs_unspiced', 'fondue', 'sauce_mushroom', 'strawberries', 'pie_plum_baked_with_cake_dough', 'potatoes_au_gratin_dauphinois_prepared', 'capers', 'bread_wholemeal_toast', 'red_radish', 'fruit_tart', 'beans_kidney', 'sauerkraut', 'mustard', 'country_fries', 'ketchup', 'pasta_linguini_parpadelle_tagliatelle', 'chicken_cut_into_stripes_only_meat', 'cookies', 'sun_dried_tomatoe', 'bread_ticino_ch', 'semi_hard_cheese', 'porridge_prepared_with_partially_skimmed_milk', 'juice', 'chocolate_milk', 'bread_fruit', 'corn', 'dates', 'pistachio', 'cream_cheese_n_s', 'bread_rye', 'witloof_chicory', 'goat_cheese_soft', 'grapefruit_pomelo', 'blue_mould_cheese', 'guacamole', 'tofu', 'cordon_bleu', 'quinoa', 'kefir_drink', 'salad_rocket', 'pizza_with_ham_with_mushrooms_baked', 'fruit_coulis', 'plums', 'pizza_with_ham_baked', 'pineapple', 'seeds_n_s', 'focaccia', 'mixed_milk_beverage', 'coleslaw_chopped_without_sauce', 'sweet_potato', 'chicken_leg', 'croissant', 'cheesecake', 'sauce_cocktail', 'croissant_with_chocolate_filling', 'pumpkin_seeds', 'artichoke', 'soft_drink_with_a_taste', 'apple_pie', 'white_bread_with_butter_eggs_and_milk', 'savoury_pastry_stick', 'tuna_in_oil_drained', 'meat_terrine_pate', 'falafel_balls', 'berries_n_s', 'latte_macchiato', 'sugar_melon_galia_honeydew_cantaloupe', 'mixed_seeds_n_s', 'oil_vinegar_salad_dressing', 'celeriac', 'chocolate_mousse', 'lemon', 'chocolate_cookies', 'birchermuesli_prepared_no_sugar_added', 'muffin', 'pine_nuts', 'french_pizza_from_alsace_baked', 'chocolate_n_s', 'grits_polenta_maize_flour', 'wine_rose', 'cola_based_drink', 'raspberries', 'roll_with_pieces_of_chocolate', 'cake_lemon', 'rice_wild', 'gluten_free_bread', 'pearl_onion', 'tzatziki', 'ham_croissant_ch', 'corn_crisps', 'lentils_green_du_puy_du_berry', 'rice_whole_grain', 'cervelat_ch', 'aperitif_with_alcohol_n_s_aperol_spritz', 'peas', 'tiramisu', 'apricots', 'lasagne_meat_prepared', 'brioche', 'vegetable_au_gratin_baked', 'basil', 'butter_spread_puree_almond', 'pie_apricot', 'rusk_wholemeal', 'pasta_in_conch_form', 'pasta_in_butterfly_form_farfalle', 'damson_plum', 'shoots_n_s', 'coconut', 'banana_cake', 'sauce_curry', 'watermelon_fresh', 'white_asparagus', 'cherries', 'nectarine')

        # # 498
        # return ('bread-wholemeal', 'jam', 'water', 'bread-sourdough', 'banana', 'soft-cheese', 'ham-raw', 'hard-cheese', 'cottage-cheese', 'bread-half-white', 'coffee-with-caffeine', 'fruit-salad', 'pancakes', 'tea', 'salmon-smoked', 'avocado', 'spring-onion-scallion', 'ristretto-with-caffeine', 'ham', 'egg', 'bacon-frying', 'chips-french-fries', 'juice-apple', 'chicken', 'tomato-raw', 'broccoli', 'shrimp-boiled', 'beetroot-steamed-without-addition-of-salt', 'carrot-raw', 'chickpeas', 'french-salad-dressing', 'pasta-hornli', 'sauce-cream', 'meat-balls', 'pasta', 'tomato-sauce', 'cheese', 'pear', 'cashew-nut', 'almonds', 'lentils', 'mixed-vegetables', 'peanut-butter', 'apple', 'blueberries', 'cucumber', 'cocoa-powder', 'greek-yaourt-yahourt-yogourt-ou-yoghourt', 'maple-syrup-concentrate', 'buckwheat-grain-peeled', 'butter', 'herbal-tea', 'mayonnaise', 'soup-vegetable', 'wine-red', 'wine-white', 'green-bean-steamed-without-addition-of-salt', 'sausage', 'pizza-margherita-baked', 'salami', 'mushroom', 'bread-meat-substitute-lettuce-sauce', 'tart', 'tea-verveine', 'rice', 'white-coffee-with-caffeine', 'linseeds', 'sunflower-seeds', 'ham-cooked', 'bell-pepper-red-raw', 'zucchini', 'green-asparagus', 'tartar-sauce', 'lye-pretzel-soft', 'cucumber-pickled', 'curry-vegetarian', 'yaourt-yahourt-yogourt-ou-yoghourt-natural', 'soup-of-lentils-dahl-dhal', 'soup-cream-of-vegetables', 'balsamic-vinegar', 'salmon', 'salt-cake-vegetables-filled', 'bacon', 'orange', 'pasta-noodles', 'cream', 'cake-chocolate', 'pasta-spaghetti', 'black-olives', 'parmesan', 'spaetzle', 'salad-lambs-ear', 'salad-leaf-salad-green', 'potatoes-steamed', 'white-cabbage', 'halloumi', 'beetroot-raw', 'bread-grain', 'applesauce-unsweetened-canned', 'cheese-for-raclette', 'mushrooms', 'bread-white', 'curds-natural-with-at-most-10-fidm', 'bagel-without-filling', 'quiche-with-cheese-baked-with-puff-pastry', 'soup-potato', 'bouillon-vegetable', 'beef-sirloin-steak', 'taboule-prepared-with-couscous', 'eggplant', 'bread', 'turnover-with-meat-small-meat-pie-empanadas', 'mungbean-sprouts', 'mozzarella', 'pasta-penne', 'lasagne-vegetable-prepared', 'mandarine', 'kiwi', 'french-beans', 'tartar-meat', 'spring-roll-fried', 'pork-chop', 'caprese-salad-tomato-mozzarella', 'leaf-spinach', 'roll-of-half-white-or-white-flour-with-large-void', 'pasta-ravioli-stuffing', 'omelette-plain', 'tuna', 'dark-chocolate', 'sauce-savoury', 'dried-raisins', 'ice-tea', 'kaki', 'macaroon', 'smoothie', 'crepe-plain', 'chicken-nuggets', 'chili-con-carne-prepared', 'veggie-burger', 'cream-spinach', 'cod', 'chinese-cabbage', 'hamburger-bread-meat-ketchup', 'soup-pumpkin', 'sushi', 'chestnuts', 'coffee-decaffeinated', 'sauce-soya', 'balsamic-salad-dressing', 'pasta-twist', 'bolognaise-sauce', 'leek', 'fajita-bread-only', 'potato-gnocchi', 'beef-cut-into-stripes-only-meat', 'rice-noodles-vermicelli', 'tea-ginger', 'tea-green', 'bread-whole-wheat', 'onion', 'garlic', 'hummus', 'pizza-with-vegetables-baked', 'beer', 'glucose-drink-50g', 'chicken-wing', 'ratatouille', 'peanut', 'high-protein-pasta-made-of-lentils-peas', 'cauliflower', 'quiche-with-spinach-baked-with-cake-dough', 'green-olives', 'brazil-nut', 'eggplant-caviar', 'bread-pita', 'pasta-wholemeal', 'sauce-pesto', 'oil', 'couscous', 'sauce-roast', 'prosecco', 'crackers', 'bread-toast', 'shrimp-prawn-small', 'panna-cotta', 'romanesco', 'water-with-lemon-juice', 'espresso-with-caffeine', 'egg-scrambled-prepared', 'juice-orange', 'ice-cubes', 'braided-white-loaf', 'emmental-cheese', 'croissant-wholegrain', 'hazelnut-chocolate-spread-nutella-ovomaltine-caotina', 'tomme', 'water-mineral', 'hazelnut', 'bacon-raw', 'bread-nut', 'black-forest-tart', 'soup-miso', 'peach', 'figs', 'beef-filet', 'mustard-dijon', 'rice-basmati', 'mashed-potatoes-prepared-with-full-fat-milk-with-butter', 'dumplings', 'pumpkin', 'swiss-chard', 'red-cabbage', 'spinach-raw', 'naan-indien-bread', 'chicken-curry-cream-coconut-milk-curry-spices-paste', 'crunch-muesli', 'biscuits', 'bread-french-white-flour', 'meatloaf', 'fresh-cheese', 'honey', 'vegetable-mix-peas-and-carrots', 'parsley', 'brownie', 'dairy-ice-cream', 'tea-black', 'carrot-cake', 'fish-fingers-breaded', 'salad-dressing', 'dried-meat', 'chicken-breast', 'mixed-salad-chopped-without-sauce', 'feta', 'praline', 'tea-peppermint', 'walnut', 'potato-salad-with-mayonnaise-yogurt-dressing', 'kebab-in-pita-bread', 'kolhrabi', 'alfa-sprouts', 'brussel-sprouts', 'bacon-cooking', 'gruyere', 'bulgur', 'grapes', 'pork-escalope', 'chocolate-egg-small', 'cappuccino', 'zucchini-stewed-without-addition-of-fat-without-addition-of-salt', 'crisp-bread-wasa', 'bread-black', 'perch-fillets-lake', 'rosti', 'mango', 'sandwich-ham-cheese-and-butter', 'muesli', 'spinach-steamed-without-addition-of-salt', 'fish', 'risotto-without-cheese-cooked', 'milk-chocolate-with-hazelnuts', 'cake-oblong', 'crisps', 'pork', 'pomegranate', 'sweet-corn-canned', 'flakes-oat', 'greek-salad', 'cantonese-fried-rice', 'sesame-seeds', 'bouillon', 'baked-potato', 'fennel', 'meat', 'bread-olive', 'croutons', 'philadelphia', 'mushroom-average-stewed-without-addition-of-fat-without-addition-of-salt', 'bell-pepper-red-stewed-without-addition-of-fat-without-addition-of-salt', 'white-chocolate', 'mixed-nuts', 'breadcrumbs-unspiced', 'fondue', 'sauce-mushroom', 'tea-spice', 'strawberries', 'tea-rooibos', 'pie-plum-baked-with-cake-dough', 'potatoes-au-gratin-dauphinois-prepared', 'capers', 'vegetables', 'bread-wholemeal-toast', 'red-radish', 'fruit-tart', 'beans-kidney', 'sauerkraut', 'mustard', 'country-fries', 'ketchup', 'pasta-linguini-parpadelle-tagliatelle', 'chicken-cut-into-stripes-only-meat', 'cookies', 'sun-dried-tomatoe', 'bread-ticino', 'semi-hard-cheese', 'margarine', 'porridge-prepared-with-partially-skimmed-milk', 'soya-drink-soy-milk', 'juice-multifruit', 'popcorn-salted', 'chocolate-filled', 'milk-chocolate', 'bread-fruit', 'mix-of-dried-fruits-and-nuts', 'corn', 'tete-de-moine', 'dates', 'pistachio', 'celery', 'white-radish', 'oat-milk', 'cream-cheese', 'bread-rye', 'witloof-chicory', 'apple-crumble', 'goat-cheese-soft', 'grapefruit-pomelo', 'risotto-with-mushrooms-cooked', 'blue-mould-cheese', 'biscuit-with-butter', 'guacamole', 'pecan-nut', 'tofu', 'cordon-bleu-from-pork-schnitzel-fried', 'paprika-chips', 'quinoa', 'kefir-drink', 'm-m-s', 'salad-rocket', 'bread-spelt', 'pizza-with-ham-with-mushrooms-baked', 'fruit-coulis', 'plums', 'beef-minced-only-meat', 'pizza-with-ham-baked', 'pineapple', 'soup-tomato', 'cheddar', 'tea-fruit', 'rice-jasmin', 'seeds', 'focaccia', 'milk', 'coleslaw-chopped-without-sauce', 'pastry-flaky', 'curd', 'savoury-puff-pastry-stick', 'sweet-potato', 'chicken-leg', 'croissant', 'sour-cream', 'ham-turkey', 'processed-cheese', 'fruit-compotes', 'cheesecake', 'pasta-tortelloni-stuffing', 'sauce-cocktail', 'croissant-with-chocolate-filling', 'pumpkin-seeds', 'artichoke', 'champagne', 'grissini', 'sweets-candies', 'brie', 'wienerli-swiss-sausage', 'syrup-diluted-ready-to-drink', 'apple-pie', 'white-bread-with-butter-eggs-and-milk', 'savoury-puff-pastry', 'anchovies', 'tuna-in-oil-drained', 'lemon-pie', 'meat-terrine-pate', 'coriander', 'falafel-balls', 'berries', 'latte-macchiato-with-caffeine', 'faux-mage-cashew-vegan-chers', 'beans-white', 'sugar-melon', 'mixed-seeds', 'hamburger', 'hamburger-bun', 'oil-vinegar-salad-dressing', 'soya-yaourt-yahourt-yogourt-ou-yoghourt', 'chocolate-milk-chocolate-drink', 'celeriac', 'chocolate-mousse', 'cenovis-yeast-spread', 'thickened-cream-35', 'meringue', 'lamb-chop', 'shrimp-prawn-large', 'beef', 'lemon', 'croque-monsieur', 'chives', 'chocolate-cookies', 'birchermuesli-prepared-no-sugar-added', 'fish-crunchies-battered', 'muffin', 'savoy-cabbage-steamed-without-addition-of-salt', 'pine-nuts', 'chorizo', 'chia-grains', 'frying-sausage', 'french-pizza-from-alsace-baked', 'chocolate', 'cooked-sausage', 'grits-polenta-maize-flour', 'gummi-bears-fruit-jellies-jelly-babies-with-fruit-essence', 'wine-rose', 'coca-cola', 'raspberries', 'roll-with-pieces-of-chocolate', 'goat-average-raw', 'lemon-cake', 'coconut-milk', 'rice-wild', 'gluten-free-bread', 'pearl-onions', 'buckwheat-pancake', 'bread-5-grain', 'light-beer', 'sugar-glazing', 'tzatziki', 'butter-herb', 'ham-croissant', 'corn-crisps', 'lentils-green-du-puy-du-berry', 'cocktail', 'rice-whole-grain', 'veal-sausage', 'cervelat', 'sorbet', 'aperitif-with-alcohol-aperol-spritz', 'dips', 'corn-flakes', 'peas', 'tiramisu', 'apricots', 'cake-marble', 'lamb', 'lasagne-meat-prepared', 'coca-cola-zero', 'cake-salted', 'dough-puff-pastry-shortcrust-bread-pizza-dough', 'rice-waffels', 'sekt', 'brioche', 'vegetable-au-gratin-baked', 'mango-dried', 'processed-meat-charcuterie', 'mousse', 'sauce-sweet-sour', 'basil', 'butter-spread-puree-almond', 'pie-apricot-baked-with-cake-dough', 'rusk-wholemeal', 'beef-roast', 'vanille-cream-cooked-custard-creme-dessert', 'pasta-in-conch-form', 'nuts', 'sauce-carbonara', 'fig-dried', 'pasta-in-butterfly-form-farfalle', 'minced-meat', 'carrot-steamed-without-addition-of-salt', 'ebly', 'damson-plum', 'shoots', 'bouquet-garni', 'coconut', 'banana-cake', 'waffle', 'apricot-dried', 'sauce-curry', 'watermelon-fresh', 'sauce-sweet-salted-asian', 'pork-roast', 'blackberry', 'smoked-cooked-sausage-of-pork-and-beef-meat-sausag', 'bean-seeds', 'italian-salad-dressing', 'white-asparagus', 'pie-rhubarb-baked-with-cake-dough', 'tomato-stewed-without-addition-of-fat-without-addition-of-salt', 'cherries', 'nectarine')